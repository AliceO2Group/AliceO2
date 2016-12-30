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

#ifndef ALICEO2_SHAREDMEMORY_
#define ALICEO2_SHAREDMEMORY_
#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/interprocess/shared_memory_object.hpp>
#include <boost/circular_buffer.hpp>
#include <boost/interprocess/sync/interprocess_mutex.hpp>
#include <boost/interprocess/sync/interprocess_condition.hpp>
#include <boost/interprocess/sync/interprocess_semaphore.hpp>
#include <boost/interprocess/sync/scoped_lock.hpp>
#include <boost/interprocess/allocators/allocator.hpp>
#include <boost/interprocess/smart_ptr/shared_ptr.hpp>
#include <boost/interprocess/smart_ptr/intrusive_ptr.hpp>
#include <boost/interprocess/smart_ptr/deleter.hpp>
#include "boost/date_time/posix_time/posix_time_types.hpp"
#include <boost/thread/thread_time.hpp>
#include <chrono>
#include <iostream>
#include <cstring>
#include <cstdlib> //std::system
#include <cstddef>
#include <cassert>
#include <utility>
#include <memory>
#include <functional>

/// boost ring buffer support in shmem
#define BOOST_CB_DISABLE_DEBUG

namespace AliceO2 {
namespace SharedMemory {

namespace bi = boost::interprocess;

class Block;
class BlockOwner;
class Manager;

using byte = unsigned char;
using HandleType = typename bi::managed_shared_memory::handle_t;
using IdType = uint32_t;

using ShmAllocatorType = bi::allocator<BlockOwner, bi::managed_shared_memory::segment_manager>;
using BufferType = boost::circular_buffer<BlockOwner, ShmAllocatorType>;
using BlockPtr = bi::intrusive_ptr<Block, bi::offset_ptr<void>>;

/// The shared memory manager. All boost magic hides behind this API
class Manager {
public:
  Manager() = delete;
  Manager(Manager&) = delete;

  ~Manager()
  {
    bool rc = bi::shared_memory_object::remove(mSegmentName.c_str());
    printf("removing segment %s, rc=%d\n", mSegmentName.c_str(), rc);
  }

  Manager(size_t size, size_t bufferSize = 1000, std::string bufferName = "mBuffer",
          std::string segmentName = "O2shmem");

  const std::string& getSegmentName() const
  {
    return mSegmentName;
  }

  /// allocate a block of shared memory including the control struct with a unique ID,
  /// return pointer to control struct
  BlockPtr allocate(size_t size) noexcept;

  /// deallocate a block descibed by Block
  /// take raw pointer to comply with deallocator interface
  /// static function can be used in deleter callbacks
  static void deallocate(Block* block);

  /// Get pointer to control block from a handle.
  /// The handle either comes from the network (e.g. sent via fairmq)
  /// or from Block::getHandle();
  BlockPtr getBlock(HandleType handle, IdType id);

  /// (de-)register the shmem block for lifetime management
  /// after registering the handle returned by Block::getHandle() will be pointing to
  /// the registered owner.
  void registerBlock(BlockPtr& block);
  void deregisterBlock(BlockPtr& block);

  // wait until some memory is consumed by any of the clients
  // return false when timeout reached
  bool waitForMemory(const boost::posix_time::time_duration& timeout);

  // produce a unique ID for a message.
  IdType getUniqueID();

  // TODO: this should be private or protected.
  bi::managed_shared_memory& Segment() const
  {
    return *(mSegment.get());
  }

private:
  std::unique_ptr<bi::managed_shared_memory> mSegment;
  std::string mSegmentName;

  // non-owning pointers
  IdType* mUniqueMessageID;
  bi::interprocess_mutex* mMutexUniqueMessageID;
  bi::interprocess_condition* mConditionNotFull;
  bi::interprocess_mutex* mMutexConditionNotFull;
  BufferType* mBuffer;
  bi::interprocess_mutex* mMutexBufferAccess;
};

/// this class the holds ownership information. In all cases a handle to an instance of this guy
/// is transferred to consumer(s)
class BlockOwner {
  friend class Manager;
  friend class Block;

public:
  BlockOwner() : mBlock{}, mID{ 0 } {};
  BlockOwner(const BlockOwner&) = default;
  BlockOwner& operator=(const BlockOwner& that) = default;
  ~BlockOwner() = default;

  // boost::interprocess::intrusive_ptr does not implement rvalue operators
  // so we need to do our own
  BlockOwner(BlockOwner&& that) : BlockOwner{}
  {
    mID = that.mID;
    mBlock.swap(that.mBlock);
    that.mID = 0;
  }

  BlockOwner& operator=(BlockOwner&& that)
  {
    if (this != &that) {
      mID = that.mID;
      mBlock.swap(that.mBlock);
      that.mID = 0;
      that.mBlock = nullptr;
    }
    return *this;
  }

  BlockOwner(Block* ptr, IdType id) : mBlock{ ptr }, mID{ id }
  {
  }

  IdType getID() const
  {
    return mID;
  }

private:
  BlockPtr mBlock;
  IdType mID;
};

/// Control block describing the shared memory region.
class Block {
  friend class BlockOwner;
  friend class Manager;

public:
  Block() = delete;
  Block(const Block&) = default;
  Block(Block&&) = default;
  Block& operator=(const Block&) = default;
  Block& operator=(Block&&) = default;
  ~Block() = default;

  using SegManType = bi::managed_shared_memory::segment_manager;
  Block(size_t nbytes, IdType id, SegManType* man, bi::interprocess_mutex* notFullMutex,
        bi::interprocess_condition* notFullCondition)
    : mSize{ nbytes },
      mUsers{ 0 },
      mSegmentManager{ man },
      mNotFullMutex{ notFullMutex },
      mNotFullCondition{ notFullCondition },
      mExternalOwner{ nullptr },
      mID{ id },
      mOwner{ this, id },
      mData{ 0 }
  {
  }

  size_t size()
  {
    return mSize;
  }

  byte* data()
  {
    return &mData[0];
  }

  // intrusive pointer boilerplate
  unsigned int use_count() const
  {
    return mUsers;
  }

  inline friend void intrusive_ptr_add_ref(Block* p)
  {
    ++p->mUsers;
  }

  inline friend void intrusive_ptr_release(Block* p)
  {
    if (--p->mUsers == 0)
      Manager::deallocate(p);
  }

  HandleType getHandle(Manager& man)
  {
    return man.Segment().get_handle_from_address((!mExternalOwner) ? static_cast<void*>(&mOwner)
                                                                   : static_cast<void*>(&*mExternalOwner));
  }

  IdType getID() const
  {
    return mID;
  }
  SegManType* getSegmentManager() const
  {
    return mSegmentManager.get();
  }

private:
  static constexpr size_t sSmallBufferSize = 1;
  size_t mSize;
  // bi::interprocess_semaphore mUsers; //use count
  unsigned int mUsers; // use count
  bi::offset_ptr<SegManType> mSegmentManager;
  bi::offset_ptr<bi::interprocess_mutex> mNotFullMutex;
  bi::offset_ptr<bi::interprocess_condition> mNotFullCondition;
  bi::offset_ptr<BlockOwner> mExternalOwner;
  IdType mID;
  BlockOwner mOwner;
  byte mData[sSmallBufferSize]; // here the data starts
};
};
};
#endif
