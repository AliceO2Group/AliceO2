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

/// \file GPUTrackingInputProvider.h
/// \author David Rohr

#ifndef GPUTRACKINGINPUTPROVIDER_H
#define GPUTRACKINGINPUTPROVIDER_H

#include "GPUDef.h"
#include "GPUProcessor.h"

namespace o2
{
namespace tpc
{
struct ClusterNative;
struct ClusterNativeAccess;
} // namespace tpc
} // namespace o2

namespace GPUCA_NAMESPACE
{
namespace gpu
{

struct GPUTrackingInOutZS;
struct GPUTPCClusterOccupancyMapBin;
class GPUTRDTrackletWord;
class GPUTRDSpacePoint;

class GPUTrackingInputProvider : public GPUProcessor
{
 public:
#ifndef GPUCA_GPUCODE
  void InitializeProcessor();
  void RegisterMemoryAllocation();
  void SetMaxData(const GPUTrackingInOutPointers& io);

  void* SetPointersTPCOccupancyMap(void* mem);
  void* SetPointersInputZS(void* mem);
  void* SetPointersInputClusterNativeAccess(void* mem);
  void* SetPointersInputClusterNativeBuffer(void* mem);
  void* SetPointersInputClusterNativeOutput(void* mem);
  void* SetPointersInputTRD(void* mem);
  void* SetPointersErrorCodes(void* mem);
#endif

  uint16_t mResourceZS = -1;
  uint16_t mResourceClusterNativeAccess = -1;
  uint16_t mResourceClusterNativeBuffer = -1;
  uint16_t mResourceClusterNativeOutput = -1;
  uint16_t mResourceErrorCodes = -1;
  uint16_t mResourceTRD = -1;
  uint16_t mResourceOccupancyMap = -1;

  bool mHoldTPCZS = false;
  bool mHoldTPCClusterNative = false;
  bool mHoldTPCClusterNativeOutput = false;
  bool mHoldTPCOccupancyMap = false;
  uint32_t mNClusterNative = 0;

  GPUTrackingInOutZS* mPzsMeta = nullptr;
  uint32_t* mPzsSizes = nullptr;
  void** mPzsPtrs = nullptr;

  uint32_t mNTRDTracklets = 0;
  bool mDoSpacepoints = false;
  uint32_t mNTRDTriggerRecords = 0;
  GPUTRDTrackletWord* mTRDTracklets = nullptr;
  GPUTRDSpacePoint* mTRDSpacePoints = nullptr;
  float* mTRDTriggerTimes = nullptr;
  int32_t* mTRDTrackletIdxFirst = nullptr;
  uint8_t* mTRDTrigRecMask = nullptr;

  o2::tpc::ClusterNativeAccess* mPclusterNativeAccess = nullptr;
  o2::tpc::ClusterNative* mPclusterNativeBuffer = nullptr;
  o2::tpc::ClusterNative* mPclusterNativeOutput = nullptr;

  uint32_t* mTPCClusterOccupancyMap = nullptr;

  uint32_t* mErrorCodes = nullptr;
};

} // namespace gpu
} // namespace GPUCA_NAMESPACE

#endif
