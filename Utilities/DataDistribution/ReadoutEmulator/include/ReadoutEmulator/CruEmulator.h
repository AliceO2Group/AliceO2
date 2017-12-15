// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef ALICEO2_CRU_EMULATOR_H_
#define ALICEO2_CRU_EMULATOR_H_

#include "ReadoutEmulator/CruMemoryHandler.h"

#include "Common/ConcurrentQueue.h"
#include "Common/ReadoutDataModel.h"

#include <Headers/DataHeader.h>

#include <stack>
#include <map>
#include <vector>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <thread>

namespace o2 {
namespace DataDistribution {

struct RawDmaChunkDesc {
  uint64_t mHBFrameID; // unused
  size_t mRawDataSize; // unused
  bool mValidHBF;
};

class CruLinkEmulator {
  static unsigned sCruUsedLinks; // keep track on already used link IDs
public:
  CruLinkEmulator(std::shared_ptr<CruMemoryHandler> pMemHandler, uint64_t pLinkBitsPerS, uint64_t pDmaChunkSize)
    : mMemHandler{ pMemHandler },
      mLinkID{ ++sCruUsedLinks },
      mLinkBitsPerS{ pLinkBitsPerS },
      mRunning{ false },
      mDmaChunkSize{ pDmaChunkSize }
  {
  }

  ~CruLinkEmulator()
  {
    stop();
  }

  void linkReadoutThread();

  /// Start "data taking" thread
  void start();
  /// Stop "data taking" thread
  void stop();

private:
  std::shared_ptr<CruMemoryHandler> mMemHandler;

  unsigned mLinkID;
  std::uint64_t mLinkBitsPerS;
  std::uint64_t mDmaChunkSize;

  std::thread mCRULinkThread;
  bool mRunning;
};
}
} /* namespace o2::DataDistribution */

#endif /* ALICEO2_CRU_EMULATOR_H_ */
