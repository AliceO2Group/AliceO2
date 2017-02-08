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

/// @file ShmSource.h
///
/// @since 2014-12-10
/// @author M. Krzewicki <mkrzewic@cern.ch>

#include <thread> // this_thread::sleep_for
#include <chrono>

#include "ShmSource/ShmSource.h"
#include "FairMQLogger.h"
#include "FairMQProgOptions.h"
#include <random>

using namespace std;
using namespace AliceO2;

ShmSource::ShmSource() : mCounter{ 0 }, mMaxSize{ 1000000 }, mDelay{ 0 }, mDisableShmem{ false }
{
}

ShmSource::~ShmSource()
{
}

void ShmSource::InitTask()
{
  mCounter = fConfig->GetValue<int>("n");
  mDelay = fConfig->GetValue<int>("sleep");
  mDisableShmem = fConfig->GetValue<bool>("disableShmem");
  mShmManager.reset(new SharedMemory::Manager{ mSharedMemorySize, 1000 });
}

bool ShmSource::ConditionalRun()
{
  if (mCounter > 0 && !--mCounter)
    return false;

  // sleep to slow things down a bit
  std::this_thread::sleep_for(std::chrono::milliseconds{ mDelay });

  // get some shared memory block
  auto msgSize = mMaxSize;

  // allocate some shared memory
  auto shmBlock = mShmManager->allocate(msgSize);

  // this loop handles the out-of-memory case, every deallocation is signalled and
  // waitForMemory() will release as soon as it can.
  while (!shmBlock) {
    if (!mShmManager->waitForMemory(boost::posix_time::seconds(10))) {
      LOG(WARN) << "timed out waiting for memory";
      return false;
    }
    shmBlock = mShmManager->allocate(msgSize);
  }

  // put some data (the id) at the beginning of the block
  if (shmBlock->size() >= sizeof(SharedMemory::IdType)) {
    *reinterpret_cast<SharedMemory::IdType*>(shmBlock->data()) = shmBlock->getID();
  }

  // check if we are publishing to one or many clients
  SharedMemoryConsumers many = SharedMemoryConsumers::One;
  if (fChannels.at("data-out").at(0).GetType().find("pub") != std::string::npos) {
    many = SharedMemoryConsumers::Many;
  }

  if (mDisableShmem) {
    many = SharedMemoryConsumers::None;
  }

  // just send normally
  Base::O2message message;
  Header::DataHeader header;

  AddMessage(message, { header }, shmBlock, many);
  Send(message, "data-out");

  return true;
}
