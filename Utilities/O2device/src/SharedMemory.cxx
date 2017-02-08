/// @copyright
/// © Copyright 2014 Copyright Holders of the ALICE O2 collaboration.
/// See https://aliceinfo.cern.ch/AliceO2 for details on the Copyright holders.
/// This software is distributed under the terms of the
/// GNU General Public License version 3 (GPL Version 3).
///
/// License text in a separate file.
///
/// In applying this license, CERN does not waive the privileges and immunities
/// granted to it by virtue of its status as an Intergovernmental Organization
/// or submit itself to any jurisdiction.

/// @headerfile SharedMemory.h
///
/// @since 2017-01-01
/// @author M. Krzewicki <mkrzewic@cern.ch>

#include "O2device/SharedMemory.h"

using namespace AliceO2::SharedMemory;

//______________________________________________________________________________
AliceO2::SharedMemory::Manager::Manager(size_t size, size_t bufferSize, std::string bufferName, std::string segmentName)
  : mSegment{ nullptr },
    mSegmentName{ segmentName },
    mUniqueMessageID{ nullptr },
    mMutexUniqueMessageID{ nullptr },
    mConditionNotFull{ nullptr },
    mMutexConditionNotFull{ nullptr },
    mBuffer{ nullptr },
    mMutexBufferAccess{ nullptr }
{
  mSegment.reset(new bi::managed_shared_memory{ bi::open_or_create, segmentName.c_str(), size });

  // the handling of unique ID generation
  mUniqueMessageID = mSegment->find_or_construct<unsigned int>("mUniqueMessageID")(0);
  mMutexUniqueMessageID = mSegment->find_or_construct<bi::interprocess_mutex>("mMutexUniqueMessageID")();

  // handling of the shm region full condition
  mConditionNotFull = mSegment->find_or_construct<bi::interprocess_condition>("mConditionNotFull")();
  mMutexConditionNotFull = mSegment->find_or_construct<bi::interprocess_mutex>("mMutexConditionNotFull")();

  // ring buffer
  mBuffer = mSegment->find_or_construct<BufferType>(bufferName.c_str())(
    bufferSize, ShmAllocatorType(mSegment->get_segment_manager()));
  mMutexBufferAccess = mSegment->find_or_construct<bi::interprocess_mutex>("mMutexBufferAccess")();
}

//______________________________________________________________________________
bool AliceO2::SharedMemory::Manager::waitForMemory(const boost::posix_time::time_duration& timeout)
{
  bi::scoped_lock<bi::interprocess_mutex> lock(*mMutexConditionNotFull);
  using namespace boost::posix_time;
  return mConditionNotFull->timed_wait(lock, ptime(boost::get_system_time() + timeout));
}

//______________________________________________________________________________
BlockPtr AliceO2::SharedMemory::Manager::allocate(size_t size) noexcept
{
  IdType id = getUniqueID();
  void* ptr = mSegment->allocate(size + sizeof(Block), std::nothrow);
  if (!ptr)
    return BlockPtr{};
  Block* block =
    new (ptr) Block{ size, id, mSegment->get_segment_manager(), mMutexConditionNotFull, mConditionNotFull };
  return BlockPtr{ block };
}

//______________________________________________________________________________
void AliceO2::SharedMemory::Manager::deallocate(Block* block)
{
  if (!block)
    return;
  auto deleter = block->mSegmentManager;
  auto fullMutex = block->mNotFullMutex;
  auto fullCondition = block->mNotFullCondition;
  block->~Block();
  deleter->deallocate(block);
  bi::scoped_lock<bi::interprocess_mutex> lock(*fullMutex);
  fullCondition->notify_all();
}

//______________________________________________________________________________
BlockPtr AliceO2::SharedMemory::Manager::getBlock(HandleType handle, IdType id)
{
  bi::scoped_lock<bi::interprocess_mutex> lock(*mMutexBufferAccess);
  BlockOwner* metaBlock = static_cast<BlockOwner*>(mSegment->get_address_from_handle(handle));
  if (!metaBlock)
    return BlockPtr{}; // maybe should throw?

  // make sure nobody deletes us before we are done:
  BlockPtr alive{ metaBlock->mBlock };
  if (!alive)
    return BlockPtr{};

  if (alive->getID() != id)
    return BlockPtr{};

  // If Block is self owned (there is no external owner, no lifetime management):
  //   - we MOVE ownership to us
  if (!alive->mExternalOwner) {
    BlockPtr moved{};
    moved.swap(metaBlock->mBlock);
    return moved;
  }

  // If the block is externally owned ( by the garbage collector):
  //   - in this case we take shared ownership
  return metaBlock->mBlock;
}

//______________________________________________________________________________
void AliceO2::SharedMemory::Manager::registerBlock(BlockPtr& block)
{
  bi::scoped_lock<bi::interprocess_mutex> lock(*mMutexBufferAccess);
  mBuffer->push_back(std::move(block->mOwner));
  block->mExternalOwner = &(mBuffer->back());
}

//______________________________________________________________________________
void AliceO2::SharedMemory::Manager::deregisterBlock(BlockPtr& block)
{
  bi::scoped_lock<bi::interprocess_mutex> lock(*mMutexBufferAccess);
  if (!block) {
    return;
  }
  if (block->mExternalOwner && block->mExternalOwner->mID == block->mID) {
    // since intrusive pointer does not implement a release(),
    // we need a "thief" to steal and lose the ownership.
    BlockOwner thief{ std::move(*block->mExternalOwner) };
    block->mExternalOwner = nullptr;
  }
}

//______________________________________________________________________________
IdType AliceO2::SharedMemory::Manager::getUniqueID()
{
  bi::scoped_lock<bi::interprocess_mutex> lock(*mMutexUniqueMessageID);
  return ++*mUniqueMessageID;
}
