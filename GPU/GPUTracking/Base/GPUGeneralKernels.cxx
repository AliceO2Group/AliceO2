// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUGeneralKernels.cxx
/// \author David Rohr

#include "GPUGeneralKernels.h"
using namespace GPUCA_NAMESPACE::gpu;

template <>
GPUdii() void GPUMemClean16::Thread<0>(int nBlocks, int nThreads, int iBlock, int iThread, GPUsharedref() MEM_LOCAL(GPUSharedMemory) & smem, processorType& processors, GPUglobalref() void* ptr, unsigned long size)
{
  const unsigned long stride = get_global_size(0);
  int4 i0;
  i0.x = i0.y = i0.z = i0.w = 0;
  int4* ptra = (int4*)ptr;
  unsigned long len = (size + sizeof(int4) - 1) / sizeof(int4);
  for (unsigned long i = get_global_id(0); i < len; i += stride) {
    ptra[i] = i0;
  }
}
