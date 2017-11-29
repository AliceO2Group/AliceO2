// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef ALICEO2_CRU_MEMORY_HANDLER_H_
#define ALICEO2_CRU_MEMORY_HANDLER_H_

#include "Common/ConcurrentQueue.h"
#include "Common/ReadoutDataModel.h"

#include "Headers/DataHeader.h"

#include <stack>
#include <map>
#include <vector>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <thread>

class FairMQUnmanagedRegion;

namespace o2 {
namespace DataDistribution {

struct CRUSuperpage {
  char* mDataVirtualAddress;
  char* mDataBusAddress;

  char* mDescVirtualAddress;
  char* mDescBusAddress;
};

struct CruDmaPacket {
  FairMQUnmanagedRegion* mDataSHMRegion = nullptr;
  char* mDataPtr = nullptr;
  size_t mDataSize = size_t(0);

  FairMQUnmanagedRegion* mDescSHMRegion = nullptr;
  char* mDescPtr = nullptr;
  size_t mDescSize = size_t(0);
};

struct ReadoutLinkO2Data {

  DataHeader mLinkDataHeader;
  std::vector<CruDmaPacket> mLinkRawData;
};

class CruMemoryHandler {
public:
  CruMemoryHandler() = default;
  ~CruMemoryHandler()
  {
    teardown();
  }

  void init(FairMQUnmanagedRegion* pDataRegion, FairMQUnmanagedRegion* pDescRegion, std::size_t pSuperPageSize,
            std::size_t pDmaChunkSize);
  void teardown();

  std::size_t getSuperpageSize() const
  {
    return mSuperpageSize;
  }

  // get a superpage from the free list
  bool getSuperpage(CRUSuperpage& sp);

  // not useful
  void put_superpage(const char* spVirtAddr);

  // address must match shm fairmq messages sent out
  void get_data_buffer(const char* dataBufferAddr, const std::size_t dataBuffSize);
  void put_data_buffer(const char* dataBufferAddr, const std::size_t dataBuffSize);
  size_t free_superpages();

  auto getDataRegion() const
  {
    return mDataRegion;
  }
  auto getDescRegion() const
  {
    return mDescRegion;
  }

  char* getDataRegionPtr() const
  {
    return static_cast<char*>(mDataRegion->GetData());
  }
  char* getDescRegionPtr() const
  {
    return static_cast<char*>(mDescRegion->GetData());
  }

  auto getDataRegionSize() const
  {
    return mDataRegion->GetSize();
  }
  auto getDescRegionSize() const
  {
    return mDescRegion->GetSize();
  }

  // fifo of filled ReadoutLinkO2Data updates to be sent to STFBuilder (thread safe)
  // linkThread<1..N> -> queue -> mCruO2InterfaceThread
  void putLinkData(ReadoutLinkO2Data&& pLinkData)
  {
    mO2LinkDataQueue.push(std::move(pLinkData));
  }
  bool getLinkData(ReadoutLinkO2Data& pLinkData)
  {
    return mO2LinkDataQueue.pop(pLinkData);
  }

private:
  FairMQUnmanagedRegion* mDataRegion;
  FairMQUnmanagedRegion* mDescRegion;

  std::size_t mSuperpageSize;

  /// stack of free superpages
  ConcurrentLifo<CRUSuperpage> mSuperpages;

  std::mutex mLock;
  std::map<const char*, CRUSuperpage> mVirtToSuperpage;
  // map<sp_address, map<buff_addr, buf_len>>
  std::map<const char*, std::map<const char*, std::size_t>> mUsedSuperPages;

  /// output data queue
  ConcurrentFifo<ReadoutLinkO2Data> mO2LinkDataQueue;
};
}
} /* namespace o2::DataDistribution */

#endif /* ALICEO2_CRU_MEMORY_HANDLER_H_ */
