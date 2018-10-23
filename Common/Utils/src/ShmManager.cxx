// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/*

 * ShmManager.cxx
 *
 *  Created on: Jun 17, 2018
 *      Author: swenzel
 */

#include "CommonUtils/ShmManager.h"
#include <FairLogger.h>
#include <sys/shm.h>
#include <sys/ipc.h>
#include <algorithm>

#include <boost/interprocess/managed_external_buffer.hpp>
#include <boost/interprocess/allocators/allocator.hpp>
#include <sstream>
#include <cstdlib>
#include <cassert>

using namespace boost::interprocess;

namespace o2
{
namespace utils
{

// the shared mem id under which this is accessible
const char* SHMIDNAME = "ALICEO2_SIMSHM_SHMID";
// a common virtual address under which this should be mapped
const char* SHMADDRNAME = "ALICEO2_SIMSHM_COMMONADDR";

ShmManager::ShmManager() {}

void* ShmManager::tryAttach(bool& success)
{
  success = false;
  if (auto id = getenv(SHMIDNAME)) {
    mShmID = atoi(getenv(SHMIDNAME));
  } else {
    mShmID = -1;
    return nullptr;
  }
  LOG(INFO) << "FOUND ID TO ATTACH " << mShmID;

  // if the segment was not created in the first place ...
  if (mShmID == -1) {
    return nullptr;
  }

  void* addr_wanted;
  if (auto addrstr = getenv(SHMADDRNAME)) {
    addr_wanted = (void*)atoll(addrstr);
  } else {
    mShmID = -1;
    return nullptr;
  }
  LOG(INFO) << "TRYING ADDRESS " << addr_wanted;
  auto addr = shmat(mShmID, addr_wanted, 0);
  if (addr != (void*)-1) {
    // this means success
    assert(addr == addr_wanted);
    success = true;
    mSegPtr = addr;
    mSegInfoPtr = static_cast<o2::utils::ShmMetaInfo*>(mSegPtr);
    return addr;
  }

  // trying without constraints
  addr = shmat(mShmID, 0, 0);
  if (addr == (void*)(-1)) {
    LOG(FATAL) << "SHOULD NOT HAPPEN";
  }
  // signal failure
  auto info = static_cast<o2::utils::ShmMetaInfo*>(addr);
  info->failures.fetch_add(1);
  shmdt(addr);
  mShmID = -1;
  mSegInfoPtr = nullptr;
  mSegPtr = nullptr;
  // otherwise for now this means that we could not map the segment
  // at the right place ...
  return nullptr;
}

void ShmManager::printSegInfo() const
{
  if (mSegInfoPtr) {
    LOG(INFO) << "ATTACHED WORKERS " << mSegInfoPtr->counter;
    LOG(INFO) << "CONNECTION FAILURES " << mSegInfoPtr->failures;
  } else {
    LOG(INFO) << "no segment info to print";
  }
}

bool ShmManager::createGlobalSegment(int nsegments)
{
  mIsMaster = true;
  // first of all take a look if we really start from a clean state
  if (mShmID != -1) {
    LOG(WARN) << "A SEGMENT IS ALREADY INITIALIZED";
    return false;
  }
  // make sure no one else has created the shm pool
  // TODO: this can be relaxed / generalized
  if (getenv(SHMIDNAME) || getenv(SHMADDRNAME)) {
    LOG(WARN) << "A SEGMET IS ALREADY PRESENT ON THE SYSTEM";
    return false;
  }

  LOG(INFO) << "CREATING SIM SHARED MEM SEGMENT FOR " << nsegments << " WORKERS";
#ifdef USESHM
  LOG(INFO) << "SIZEOF ShmMetaInfo " << sizeof(ShmMetaInfo);
  const auto totalsize = sizeof(ShmMetaInfo) + SHMPOOLSIZE * nsegments;
  if ((mShmID = shmget(IPC_PRIVATE, totalsize, IPC_CREAT | 0666)) == -1) {
    perror("shmget: shmget failed");
  } else {
    // We are attaching once to determine a common virtual address under which everyone else should attach.
    // In this case (if it succeeds) we can also share objects with shm pointers
    auto addr = shmat(mShmID, nullptr, 0);
    mSegPtr = addr;
    LOG(INFO) << "COMMON ADDRESS " << addr << " AS NUMBER " << (unsigned long long)addr;

    // initialize the meta information (counter)
    o2::utils::ShmMetaInfo info;
    std::memcpy(addr, &info, sizeof(info));
    mSegInfoPtr = static_cast<o2::utils::ShmMetaInfo*>(mSegPtr);
    mSegInfoPtr->allocedbytes = totalsize;

    // communicating information about this segment via
    // an environment variable
    // TODO: consider using named posix shared memory segments to avoid this
    setenv(SHMIDNAME, std::to_string(mShmID).c_str(), 1);
    setenv(SHMADDRNAME, std::to_string((unsigned long long)(addr)).c_str(), 1);
  }
  LOG(INFO) << "SHARED MEM INITIALIZED AT ID " << mShmID;
  if (mShmID == -1) {
    LOG(WARN) << "COULD NOT CREATE SHARED MEMORY";
    setenv(SHMIDNAME, std::to_string(mShmID).c_str(), 1);
    setenv(SHMADDRNAME, std::to_string(0).c_str(), 1);
  }
#endif
}

bool ShmManager::attachToGlobalSegment()
{
  LOG(INFO) << "OCCUPYING A SEGMENT IN A SHARED REGION";
  // get information about the global shared mem in which to occupy a region
  if (!(getenv(SHMIDNAME) && getenv(SHMADDRNAME))) {
    LOG(WARN) << "NO INFORMATION ABOUT SHARED SEGMENT FOUND";
    return false;
  }

  if (mShmID != -1) {
    LOG(WARN) << "REGION ALREADY OCCUPIED OR CREATED";
    return false;
  }

  bool b;
  tryAttach(b);
}

void ShmManager::occupySegment()
{
  LOG(INFO) << "OCCUPYING A SEGMENT IN A SHARED REGION";
  // get information about the global shared mem in which to occupy a region
  if (!(getenv(SHMIDNAME) && getenv(SHMADDRNAME))) {
    LOG(WARN) << "NO INFORMATION ABOUT SHARED SEGMENT FOUND";
    return;
  }

  if (mShmID != -1) {
    LOG(WARN) << "REGION ALREADY OCCUPIED OR CREATED";
    return;
  }

  bool b;
  auto addr = tryAttach(b);
  if (b) {
    // read meta information in the segment to determine the id of this segment
    auto info = static_cast<o2::utils::ShmMetaInfo*>(addr);
    const int segmentcounter = info->counter.fetch_add(1);
    LOG(INFO) << "SEGMENTCOUNT " << segmentcounter;

    const auto offset_in_bytes = sizeof(ShmMetaInfo) + segmentcounter * SHMPOOLSIZE;
    mBufferPtr = (void*)(((char*)addr) + offset_in_bytes);

    assert((unsigned long long)((char*)mBufferPtr - (char*)addr) + SHMPOOLSIZE <= info->allocedbytes);

    boostmanagedbuffer = new boost::interprocess::wmanaged_external_buffer(create_only, mBufferPtr, SHMPOOLSIZE);
    boostallocator = new boost::interprocess::allocator<char, wmanaged_external_buffer::segment_manager>(
      boostmanagedbuffer->get_segment_manager());

    LOG(INFO) << "SHARED MEM OCCUPIED AT ID " << mShmID << " AND SEGMENT COUNTER " << segmentcounter;
  } else {
    LOG(INFO) << "ATTACH NOT SUCCESSFUL";
  }
}

ShmManager::~ShmManager()
{
  release();
}

// This implements a very malloc/free mechanism ...
// ... but we are using available boost functionality
void* ShmManager::getmemblock(size_t size)
{
  void* addr = nullptr;
  try {
    addr = (void*)boostallocator->allocate(size).get();
  } catch (const std::exception& e) {
    LOG(FATAL) << "THROW IN BOOST SHM ALLOCATION";
  };
  return addr;
}

void ShmManager::freememblock(void* ptr, size_t s)
{
  boostallocator->deallocate((char*)ptr, s);
}

void ShmManager::release()
{
#ifdef USESHM
  if (mIsMaster) {
    LOG(INFO) << "REMOVING SHARED MEM SEGMENT ID " << mShmID;
    if (mShmID != -1) {
      shmctl(mShmID, IPC_RMID, nullptr);
      mShmID = -1;
    }
  }
#endif
}

} // end namespace utils
} // end namespace o2
