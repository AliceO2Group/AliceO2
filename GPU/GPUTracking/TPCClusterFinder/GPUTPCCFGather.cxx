// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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
GPUdii() void GPUTPCCFGather::Thread<0>(int nBlocks, int nThreads, int iBlock, int iThread, GPUSharedMemory& smem, processorType& clusterer, o2::tpc::ClusterNative* ptr)
{
  for (int i = 0; i < iBlock; i++) {
    ptr += clusterer.mPclusterInRow[i];
  }
  for (unsigned int i = iThread; i < clusterer.mPclusterInRow[iBlock]; i += nThreads) {
    ptr[i] = clusterer.mPclusterByRow[iBlock * clusterer.mNMaxClusterPerRow + i];
  }
}
