// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file DecodeZS.h
/// \author David Rohr

#ifndef O2_GPU_DECODE_ZS_H
#define O2_GPU_DECODE_ZS_H

#include "clusterFinderDefs.h"
#include "GPUGeneralKernels.h"
#include "GPUConstantMem.h"
#include "DataFormatsTPC/ZeroSuppression.h"

namespace GPUCA_NAMESPACE::gpu
{

class GPUTPCClusterFinder;

class GPUTPCCFDecodeZS : public GPUKernelTemplate
{
 public:
  struct GPUSharedMemory /*: public GPUKernelTemplate::GPUSharedMemoryScan64<int, GPUCA_WARP_SIZE>*/ {
    CA_SHARED_STORAGE(unsigned int ZSPage[o2::tpc::TPCZSHDR::TPC_ZS_PAGE_SIZE / sizeof(unsigned int)]);
    unsigned int RowClusterOffset[o2::tpc::TPCZSHDR::TPC_MAX_ZS_ROW_IN_ENDPOINT];
    unsigned int nRowsRegion;
    unsigned int regionStartRow;
    unsigned int nThreadsPerRow;
    unsigned int rowStride;
    GPUAtomic(unsigned int) rowOffsetCounter;
  };

  enum K : int {
    decodeZS,
  };

  static GPUd() void decode(GPUTPCClusterFinder& clusterer, GPUSharedMemory& s, int nBlocks, int nThreads, int iBlock, int iThread, int firstHBF);

#ifdef GPUCA_HAVE_O2HEADERS
  typedef GPUTPCClusterFinder processorType;
  GPUhdi() static processorType* Processor(GPUConstantMem& processors)
  {
    return processors.tpcClusterer;
  }
#endif

  GPUhdi() CONSTEXPR static GPUDataTypes::RecoStep GetRecoStep()
  {
    return GPUDataTypes::RecoStep::TPCClusterFinding;
  }

  template <int iKernel = defaultKernel, typename... Args>
  GPUd() static void Thread(int nBlocks, int nThreads, int iBlock, int iThread, GPUSharedMemory& smem, processorType& clusterer, Args... args);
};

} // namespace GPUCA_NAMESPACE::gpu

#endif
