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

/// \file GPUTPCCFGather.cxx
/// \author David Rohr

#include "GPUTPCCFGather.h"
using namespace GPUCA_NAMESPACE::gpu;
using namespace GPUCA_NAMESPACE::gpu::tpccf;

template <>
GPUdii() void GPUTPCCFGather::Thread<0>(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, GPUSharedMemory& smem, processorType& clusterer, o2::tpc::ClusterNative* ptr)
{
  for (int32_t i = 0; i < iBlock; i++) {
    ptr += clusterer.mPclusterInRow[i];
  }
  for (uint32_t i = iThread; i < clusterer.mPclusterInRow[iBlock]; i += nThreads) {
    ptr[i] = clusterer.mPclusterByRow[iBlock * clusterer.mNMaxClusterPerRow + i];
  }
}
