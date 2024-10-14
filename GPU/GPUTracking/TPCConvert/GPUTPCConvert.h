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

/// \file GPUTPCConvert.h
/// \author David Rohr

#ifndef GPUTPCCONVERT_H
#define GPUTPCCONVERT_H

#include "GPUDef.h"
#include "GPUProcessor.h"

namespace GPUCA_NAMESPACE
{
namespace gpu
{
struct GPUTPCClusterData;

class GPUTPCConvert : public GPUProcessor
{
  friend class GPUTPCConvertKernel;
  friend class GPUChainTracking;

 public:
#ifndef GPUCA_GPUCODE
  void InitializeProcessor();
  void RegisterMemoryAllocation();
  void SetMaxData(const GPUTrackingInOutPointers& io);

  void* SetPointersOutput(void* mem);
  void* SetPointersMemory(void* mem);
#endif

  constexpr static uint32_t NSLICES = GPUCA_NSLICES;

  struct Memory {
    GPUTPCClusterData* clusters[NSLICES];
  };

 protected:
  Memory* mMemory = nullptr;
  GPUTPCClusterData* mClusters = nullptr;
  uint32_t mNClustersTotal = 0;

  int16_t mMemoryResOutput = -1;
  int16_t mMemoryResMemory = -1;
};
} // namespace gpu
} // namespace GPUCA_NAMESPACE

#endif
