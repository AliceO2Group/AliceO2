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

namespace o2 {
namespace DataDistribution {

void CruMemoryHandler::teardown()
{
  mO2LinkDataQueue.flush(); // get will not block, return false
  mSuperpages.flush();
  std::lock_guard<std::mutex> lock(mLock);
  mVirtToSuperpage.clear();
  mUsedSuperPages.clear();
}

void CruMemoryHandler::init(FairMQUnmanagedRegion* pDataRegion, FairMQUnmanagedRegion* pDescRegion,
                            std::size_t pSuperPageSize, std::size_t pDmaChunkSize)
{
  teardown();

  mSuperpageSize = pSuperPageSize;
  mDataRegion = pDataRegion;
  mDescRegion = pDescRegion;

  const auto lCntSuperpages = getDataRegionSize() / mSuperpageSize;

  LOG(INFO) << "Initializing the segment memory. Can take a while...";
  // make sure the memory is allocated properly
  std::memset(getDataRegionPtr(), 0xDA, getDataRegionSize());
  std::memset(getDescRegionPtr(), 0xDE, getDescRegionSize());

  // lock and initialize the empty page queue
  std::lock_guard<std::mutex> lock(mLock);

  mSuperpages.flush();
  mVirtToSuperpage.clear();
  mUsedSuperPages.clear();

  for (size_t i = 0; i < lCntSuperpages; i++) {
    const CRUSuperpage sp{
      getDataRegionPtr() + (i * mSuperpageSize),                                           nullptr,
      getDescRegionPtr() + (i * mSuperpageSize / pDmaChunkSize * sizeof(RawDmaChunkDesc)), nullptr
    };

    // stack of free superpages to feed the CRU
    mSuperpages.push(sp);
    // Virtual address to superpage mapping to help with returning of the used pages
    mVirtToSuperpage[sp.mDataVirtualAddress] = sp;
  }

  LOG(INFO) << "CRU Memory Handler initialization finished. Using " << lCntSuperpages << " superpages";
}

bool CruMemoryHandler::getSuperpage(CRUSuperpage& sp)
{
  return mSuperpages.try_pop(sp);
}

void CruMemoryHandler::put_superpage(const char* spVirtAddr)
{
  std::lock_guard<std::mutex> lock(mLock); // needed for the mVirtToSuperpage[] lookup
  mSuperpages.push(mVirtToSuperpage[spVirtAddr]);
}

size_t CruMemoryHandler::free_superpages()
{
  return mSuperpages.size();
}

void CruMemoryHandler::get_data_buffer(const char* dataBufferAddr, const std::size_t dataBuffSize)
{
  const char* spStartAddr = reinterpret_cast<char*>((uintptr_t)dataBufferAddr & ~((uintptr_t)mSuperpageSize - 1));

  std::lock_guard<std::mutex> lock(mLock);

  // make sure the data buffer is not already in use
  if (mUsedSuperPages[spStartAddr].count(dataBufferAddr) != 0) {
    LOG(ERROR) << "Data buffer is already in the used list! " << std::hex << (uintptr_t)dataBufferAddr << std::dec;
    return;
  }

  mUsedSuperPages[spStartAddr][dataBufferAddr] = dataBuffSize;
}

void CruMemoryHandler::put_data_buffer(const char* dataBufferAddr, const std::size_t dataBuffSize)
{
  const char* spStartAddr = reinterpret_cast<char*>((uintptr_t)dataBufferAddr & ~((uintptr_t)mSuperpageSize - 1));

  if (spStartAddr < getDataRegionPtr() || spStartAddr > getDataRegionPtr() + getDataRegionSize()) {
    LOG(ERROR) << "Returned data buffer outside of the data segment! " << std::hex
               << reinterpret_cast<uintptr_t>(spStartAddr) << " " << reinterpret_cast<uintptr_t>(dataBufferAddr) << " "
               << reinterpret_cast<uintptr_t>(getDataRegionPtr()) << " "
               << reinterpret_cast<uintptr_t>(getDataRegionPtr() + getDataRegionSize()) << std::dec
               << "(sp, in, base, last)";
    return;
  }

  std::lock_guard<std::mutex> lock(mLock);

  if (mUsedSuperPages.count(spStartAddr) == 0) {
    LOG(ERROR) << "Returned data buffer is not in the list of used superpages!";
    return;
  }

  auto& spBuffMap = mUsedSuperPages[spStartAddr];

  if (spBuffMap.count(dataBufferAddr) == 0) {
    LOG(ERROR) << "Returned data buffer is not marked as used within the superpage!";
    return;
  }

  if (spBuffMap[dataBufferAddr] != dataBuffSize) {
    LOG(ERROR) << "Returned data buffer size does not match the records: " << spBuffMap[dataBufferAddr]
               << " != " << dataBuffSize << "(recorded != returned)";
    return;
  }

  if (spBuffMap.size() > 1) {
    spBuffMap.erase(dataBufferAddr);
  } else if (spBuffMap.size() == 1) {
    mUsedSuperPages.erase(spStartAddr);
    mSuperpages.push(mVirtToSuperpage[spStartAddr]);
  } else {
    LOG(ERROR) << "Superpage chunk lost.";
  }
}
}
} /* namespace o2::DataDistribution */
