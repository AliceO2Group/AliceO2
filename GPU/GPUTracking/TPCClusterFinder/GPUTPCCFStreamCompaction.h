// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file StreamCompaction.h
/// \author Felix Weiglhofer

#ifndef O2_GPU_STREAM_COMPACTION_H
#define O2_GPU_STREAM_COMPACTION_H

#include "clusterFinderDefs.h"
#include "GPUGeneralKernels.h"
#include "GPUConstantMem.h"
#include "GPUTPCClusterFinder.h"

namespace GPUCA_NAMESPACE::gpu
{

class GPUTPCCFStreamCompaction : public GPUKernelTemplate
{

 public:
  enum K : int {
    scanStart = 0,
    scanUp = 1,
    scanTop = 2,
    scanDown = 3,
    compactDigits = 4,
  };

  struct GPUSharedMemory : public GPUKernelTemplate::GPUSharedMemoryScan64<int, GPUCA_THREAD_COUNT_SCAN> {
  };

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

  template <int iKernel = GPUKernelTemplate::defaultKernel, typename... Args>
  GPUd() static void Thread(int nBlocks, int nThreads, int iBlock, int iThread, GPUSharedMemory& smem, processorType& clusterer, Args... args);

 private:
  static GPUd() void nativeScanUpStartImpl(int, int, int, int, GPUSharedMemory&,
                                           const uchar*, int*, int*,
                                           int);

  static GPUd() void nativeScanUpImpl(int, int, int, int, GPUSharedMemory&,
                                      int*, int*, int);

  static GPUd() void nativeScanTopImpl(int, int, int, int, GPUSharedMemory&,
                                       int*, int);

  static GPUd() void nativeScanDownImpl(int, int, int, int, GPUSharedMemory&,
                                        int*, const int*, unsigned int, int);

  static GPUd() void compactImpl(int, int, int, int, GPUSharedMemory&,
                                 const ChargePos*, ChargePos*,
                                 const uchar*, int*, const int*,
                                 int, tpccf::SizeT);
  static GPUd() int compactionElems(processorType& clusterer, int stage);
};

} // namespace GPUCA_NAMESPACE::gpu

#endif
