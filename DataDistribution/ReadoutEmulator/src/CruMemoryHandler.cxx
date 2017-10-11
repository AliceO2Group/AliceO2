// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "ReadoutEmulator/CruMemoryHandler.h"
#include "ReadoutEmulator/CruEmulator.h"
#include "Common/ConcurrentQueue.h"

#include <fairmq/FairMQUnmanagedRegion.h>
#include <fairmq/FairMQDevice.h> /* NewUnmanagedRegionFor */
#include <options/FairMQProgOptions.h>
#include <FairMQLogger.h>

#include <chrono>
#include <thread>

namespace o2
{
namespace DataDistribution
{

constexpr unsigned long CruMemoryHandler::cBufferBucketSize;

void CruMemoryHandler::teardown()
{
  mO2LinkDataQueue.stop(); // get will not block, return false
  mSuperpages.stop();

  for (auto b = 0; b < cBufferBucketSize; b++) {
    std::lock_guard<std::mutex> lock(mBufferMap[b].mLock);
    mBufferMap[b].mVirtToSuperpage.clear();
    mBufferMap[b].mUsedSuperPages.clear();
  }
}

void CruMemoryHandler::init(FairMQUnmanagedRegion* pDataRegion, std::size_t pSuperPageSize, std::size_t pDmaChunkSize)
{
  mSuperpageSize = pSuperPageSize;
  mDataRegion = pDataRegion;

  const auto lCntSuperpages = getDataRegionSize() / mSuperpageSize;

  LOG(INFO) << "Initializing the segment memory. Can take a while...";
  // make sure the memory is allocated properly
  {
    char* lPtr = getDataRegionPtr();
    for (std::size_t i = 0; i < getDataRegionSize(); i++) {
      lPtr[i] = i;
    }
  }

  // lock and initialize the empty page queue
  mSuperpages.flush();

  for (auto b = 0; b < cBufferBucketSize; b++) {
    std::lock_guard<std::mutex> lock(mBufferMap[b].mLock);
    mBufferMap[b].mVirtToSuperpage.clear();
    mBufferMap[b].mUsedSuperPages.clear();
  }

  for (size_t i = 0; i < lCntSuperpages; i++) {
    const CRUSuperpage sp{ getDataRegionPtr() + (i * mSuperpageSize), nullptr };
    // stack of free superpages to feed the CRU
    mSuperpages.push(sp);

    // Virtual address to superpage mapping to help with returning of the used pages
    auto& lBucket = getBufferBucket(sp.mDataVirtualAddress);
    std::lock_guard<std::mutex> lock(lBucket.mLock);
    lBucket.mVirtToSuperpage[sp.mDataVirtualAddress] = sp;
  }

  LOG(INFO) << "CRU Memory Handler initialization finished. Using " << lCntSuperpages << " superpages";
}

bool CruMemoryHandler::getSuperpage(CRUSuperpage& sp)
{
  return mSuperpages.try_pop(sp);
}

void CruMemoryHandler::put_superpage(const char* spVirtAddr)
{
  auto& lBucket = getBufferBucket(spVirtAddr);

  std::lock_guard<std::mutex> lock(lBucket.mLock); // needed for the mVirtToSuperpage[] lookup
  mSuperpages.push(lBucket.mVirtToSuperpage[spVirtAddr]);
}

size_t CruMemoryHandler::free_superpages()
{
  return mSuperpages.size();
}

void CruMemoryHandler::get_data_buffer(const char* dataBufferAddr, const std::size_t dataBuffSize)
{
  const char* lSpStartAddr = reinterpret_cast<char*>((uintptr_t)dataBufferAddr & ~((uintptr_t)mSuperpageSize - 1));

  auto& lBucket = getBufferBucket(lSpStartAddr);
  std::lock_guard<std::mutex> lock(lBucket.mLock);

  // make sure the data buffer is not already in use
  if (lBucket.mUsedSuperPages[lSpStartAddr].count(dataBufferAddr) != 0) {
    LOG(ERROR) << "Data buffer is already in the used list! " << std::hex << (uintptr_t)dataBufferAddr << std::dec;
    return;
  }

  lBucket.mUsedSuperPages[lSpStartAddr][dataBufferAddr] = dataBuffSize;
}

void CruMemoryHandler::put_data_buffer(const char* dataBufferAddr, const std::size_t dataBuffSize)
{
  const char* lSpStartAddr = reinterpret_cast<char*>((uintptr_t)dataBufferAddr & ~((uintptr_t)mSuperpageSize - 1));

  if (lSpStartAddr < getDataRegionPtr() || lSpStartAddr > getDataRegionPtr() + getDataRegionSize()) {
    LOG(ERROR) << "Returned data buffer outside of the data segment! " << std::hex
               << reinterpret_cast<uintptr_t>(lSpStartAddr) << " " << reinterpret_cast<uintptr_t>(dataBufferAddr) << " "
               << reinterpret_cast<uintptr_t>(getDataRegionPtr()) << " "
               << reinterpret_cast<uintptr_t>(getDataRegionPtr() + getDataRegionSize()) << std::dec
               << "(sp, in, base, last)";
    return;
  }

  const auto lDataBufferAddr = dataBufferAddr;
  const auto lDataBuffSize = dataBuffSize;

  auto& lBucket = getBufferBucket(lSpStartAddr);
  std::lock_guard<std::mutex> lock(lBucket.mLock);

  if (lBucket.mUsedSuperPages.count(lSpStartAddr) == 0) {
    LOG(ERROR) << "Returned data buffer is not in the list of used superpages!";
    return;
  }

  auto& lSpBuffMap = lBucket.mUsedSuperPages[lSpStartAddr];

  if (lSpBuffMap.count(lDataBufferAddr) == 0) {
    LOG(ERROR) << "Returned data buffer is not marked as used within the superpage!";
    return;
  }

  if (lSpBuffMap[lDataBufferAddr] != lDataBuffSize) {
    LOG(ERROR) << "Returned data buffer size does not match the records: " << lSpBuffMap[lDataBufferAddr]
               << " != " << lDataBuffSize << "(recorded != returned)";
    return;
  }

  if (lSpBuffMap.size() > 1) {
    lSpBuffMap.erase(lDataBufferAddr);
  } else if (lSpBuffMap.size() == 1) {
    lBucket.mUsedSuperPages.erase(lSpStartAddr);
    mSuperpages.push(lBucket.mVirtToSuperpage[lSpStartAddr]);
  } else {
    LOG(ERROR) << "Superpage chunk lost.";
  }
}
}
} /* namespace o2::DataDistribution */
