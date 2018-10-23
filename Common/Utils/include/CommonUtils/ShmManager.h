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
 * ShmManager.h
 *
 *  Created on: Jun 17, 2018
 *      Author: swenzel
 */

#ifndef COMMON_UTILS_INCLUDE_COMMONUTILS_SHMMANAGER_H_
#define COMMON_UTILS_INCLUDE_COMMONUTILS_SHMMANAGER_H_

#include <list>
#include <cstddef>

#include <boost/interprocess/managed_external_buffer.hpp>
#include <boost/interprocess/allocators/allocator.hpp>

#define USESHM 1

namespace o2
{
namespace utils
{

// the size dedicated to each attached worker/process
constexpr size_t SHMPOOLSIZE = 1024 * 1024 * 200; // 200MB

// some meta info stored at the beginning of the global shared mem segment
struct ShmMetaInfo {
  unsigned long long allocedbytes = 0;
  std::atomic<int> counter = 0; // atomic counter .. counter number of attached processes
                                // and used to assign a subregion to the attached processes
  std::atomic<int> failures = 0;
};

// Class creating -- or attaching to -- a shared memory pool
// and manages allocations within the pool
// This is used in the parallel simulation in order
// to put hits directly in shared mem; I hope this can be replaced/refactored
// to use directly functionality by FairMQ some day.
// For the moment a wrapper around boost allocators ... enhancing them with some state.
class ShmManager
{
 public:
  static ShmManager& Instance()
  {
    static ShmManager instance;
    return instance;
  }

  // creates a global shared mem region
  // to be used by "nsubsegments" simulation processes
  bool createGlobalSegment(int nsubsegments = 1);

  // create the local segment
  // this will occupy a subregion of an already created global shared mem segment
  void occupySegment();

  // simply attaches to the global segment
  bool attachToGlobalSegment();

  // the equivalent of malloc
  void* getmemblock(size_t size);
  // the equivalent of free
  void freememblock(void*, std::size_t = 1);

  void release();
  int getShmID() const { return mShmID; }
  bool hasSegment() const { return mShmID != -1; }
  bool readyToAllocate() const { return mShmID != -1 && mBufferPtr; }

  // returns if pointer is part of the shm region under control of this manager
  bool isPointerOk(void* ptr) const
  {
    return mBufferPtr && getPointerOffset(ptr) < SHMPOOLSIZE;
  }

  // returns if shared mem setup is correctly setup/operational
  // used to decide whether to communicate via shared mem at runtime or via
  // TMessages /etc/
  bool isOperational() const { return mSegInfoPtr && mSegInfoPtr->failures == 0; /* mIsOperational; */ }

  void disable()
  {
    if (mSegInfoPtr) {
      mSegInfoPtr->failures.fetch_add(1);
    };
  }

  void printSegInfo() const;

 private:
  ShmManager();
  ~ShmManager();
  int mShmID = -1;                    // id of shared mem created or used
  void* mBufferPtr = nullptr;         // the mapped/start ptr of the buffer to use
  void* mSegPtr = nullptr;            // address of the segment start
  ShmMetaInfo* mSegInfoPtr = nullptr; // pointing to the meta information object
  bool mIsMaster = false;             // true if the manager who allocated the region
  bool mIsOperational = false;
  // helper function
  void* tryAttach(bool& success);
  size_t getPointerOffset(void* ptr) const { return (size_t)((char*)ptr - (char*)mBufferPtr); }

  boost::interprocess::wmanaged_external_buffer* boostmanagedbuffer;
  boost::interprocess::allocator<char, boost::interprocess::wmanaged_external_buffer::segment_manager>* boostallocator;
};

} // namespace utils
} // namespace o2

#endif /* COMMON_UTILS_INCLUDE_COMMONUTILS_SHMMANAGER_H_ */
