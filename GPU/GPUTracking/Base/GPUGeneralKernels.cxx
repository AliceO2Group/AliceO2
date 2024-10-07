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

/// \file GPUGeneralKernels.cxx
/// \author David Rohr

#include "GPUGeneralKernels.h"
#include "GPUConstantMem.h"
using namespace GPUCA_NAMESPACE::gpu;

template <>
GPUdii() void GPUMemClean16::Thread<0>(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, GPUsharedref() MEM_LOCAL(GPUSharedMemory) & smem, processorType& GPUrestrict() processors, GPUglobalref() void* ptr, uint64_t size)
{
  const uint64_t stride = get_global_size(0);
  int4 i0;
  i0.x = i0.y = i0.z = i0.w = 0;
  int4* ptra = (int4*)ptr;
  uint64_t len = (size + sizeof(int4) - 1) / sizeof(int4);
  for (uint64_t i = get_global_id(0); i < len; i += stride) {
    ptra[i] = i0;
  }
}

template <>
GPUdii() void GPUitoa::Thread<0>(int32_t nBlocks, int32_t nThreads, int32_t iBlock, int32_t iThread, GPUsharedref() MEM_LOCAL(GPUSharedMemory) & smem, processorType& GPUrestrict() processors, GPUglobalref() int32_t* ptr, uint64_t size)
{
  const uint64_t stride = get_global_size(0);
  for (uint64_t i = get_global_id(0); i < size; i += stride) {
    ptr[i] = i;
  }
}
