// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUChainTrackingDEBUG.h
/// \author David Rohr

#ifndef GPUCHAINTRACKINGDEBUG_H
#define GPUCHAINTRACKINGDEBUG_H

#include <cstdint>
#include <functional>
#include <fstream>

namespace o2::gpu
{
// NOTE: Values below 262144 are activated by default with --debug 6 in GPUSettingsList.h::debugMask
enum GPUChainTrackingDebugFlags : uint32_t {
  TPCSectorTrackingData = 1 << 0,
  TPCPreLinks = 1 << 1,
  TPCLinks = 1 << 2,
  TPCStartHits = 1 << 3,
  TPCTracklets = 1 << 4,
  TPCHitWeights = 1 << 5,
  TPCSectorTracks = 1 << 6,
  TPCMergingRanges = 1 << 7,
  TPCMergingSectorTracks = 1 << 8,
  TPCMergingMatching = 1 << 9,
  TPCMergingCollectedTracks = 1 << 10,
  TPCMergingCE = 1 << 11,
  TPCMergingPrepareFit = 1 << 12,
  TPCMergingRefit = 1 << 13,
  TPCMergingLoopers = 1 << 14,
  TPCCompressedClusters = 1 << 15,
  TPCDecompressedClusters = 1 << 16,
  TPCClustererClusters = 1 << 17,
  TPCClustererDigits = 1 << 18,
  TPCClustererPeaks = 1 << 19,
  TPCClustererSuppressedPeaks = 1 << 20,
  TPCClustererChargeMap = 1 << 21,
  TPCClustererZeroedCharges = 1 << 22
};

template <class T, class S, typename... Args>
inline bool GPUChain::DoDebugAndDump(GPUChain::RecoStep step, uint32_t mask, bool transfer, T& processor, S T::*func, Args&&... args)
{
  if (GetProcessingSettings().keepAllMemory) {
    if (transfer) {
      TransferMemoryResourcesToHost(step, &processor, -1, true);
    }
    std::function<void(Args && ...)> lambda = [&processor, &func](Args&... args_tmp) {
      if (func) {
        (processor.*func)(args_tmp...);
      }
    };
    return DoDebugDump(mask, lambda, args...);
  }
  return false;
}

template <typename... Args>
inline bool GPUChain::DoDebugDump(uint32_t mask, std::function<void(Args&...)> func, Args&... args)
{
  if (GetProcessingSettings().debugLevel >= 6 && (mask == 0 || (GetProcessingSettings().debugMask & mask))) {
    func(args...);
    return true;
  }
  return false;
}

} // namespace o2::gpu

#endif
