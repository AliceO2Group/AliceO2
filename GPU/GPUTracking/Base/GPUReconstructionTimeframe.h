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

/// \file GPUReconstructionTimeframe.h
/// \author David Rohr

#ifndef GPURECONSTRUCTIONTIMEFRAME_H
#define GPURECONSTRUCTIONTIMEFRAME_H

#include "GPUChainTracking.h"
#include "GPUDataTypes.h"
#include "GPUTPCGeometry.h"
#include <vector>
#include <random>
#include <tuple>

namespace o2::tpc
{
struct ClusterNative;
} // namespace o2::tpc

namespace GPUCA_NAMESPACE
{
namespace gpu
{
struct ClusterNativeAccess;

class GPUReconstructionTimeframe
{
 public:
  GPUReconstructionTimeframe(GPUChainTracking* rec, int32_t (*read)(int32_t), int32_t nEvents);
  int32_t LoadCreateTimeFrame(int32_t iEvent);
  int32_t LoadMergedEvents(int32_t iEvent);
  int32_t ReadEventShifted(int32_t i, float shiftZ, float minZ = -1e6, float maxZ = -1e6, bool silent = false);
  void MergeShiftedEvents();

  static constexpr int32_t ORBIT_RATE = 11245;
  static constexpr int32_t DRIFT_TIME = 93000;
  static constexpr int32_t TPCZ = GPUTPCGeometry::TPCLength();
  static constexpr int32_t TIME_ORBIT = 1000000000 / ORBIT_RATE;

 private:
  constexpr static uint32_t NSLICES = GPUReconstruction::NSLICES;

  void SetDisplayInformation(int32_t iCol);

  GPUChainTracking* mChain;
  int32_t (*mReadEvent)(int32_t);
  int32_t mNEventsInDirectory;

  std::uniform_real_distribution<double> mDisUniReal;
  std::uniform_int_distribution<uint64_t> mDisUniInt;
  std::mt19937_64 mRndGen1;
  std::mt19937_64 mRndGen2;

  int32_t mTrainDist = 0;
  float mCollisionProbability = 0.f;
  int32_t mMaxBunchesFull;
  int32_t mMaxBunches;

  int32_t mNTotalCollisions = 0;

  int64_t mEventStride;
  int32_t mSimBunchNoRepeatEvent;
  std::vector<int8_t> mEventUsed;
  std::vector<std::tuple<GPUTrackingInOutPointers, GPUChainTracking::InOutMemory, o2::tpc::ClusterNativeAccess>> mShiftedEvents;
};
} // namespace gpu
} // namespace GPUCA_NAMESPACE

#endif
