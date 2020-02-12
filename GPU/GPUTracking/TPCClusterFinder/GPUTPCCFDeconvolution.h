// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file Deconvolution.h
/// \author Felix Weiglhofer

#ifndef O2_GPU_DECONVOLUTION_H
#define O2_GPU_DECONVOLUTION_H

#include "clusterFinderDefs.h"
#include "GPUGeneralKernels.h"
#include "GPUConstantMem.h"
#include "GPUTPCClusterFinder.h"
#include "Array2D.h"
#include "PackedCharge.h"

namespace GPUCA_NAMESPACE
{
namespace gpu
{

class GPUTPCCFDeconvolution : public GPUKernelTemplate
{

 public:
  struct GPUSharedMemory : public GPUKernelTemplate::GPUSharedMemoryScan64<short, GPUCA_THREAD_COUNT_CLUSTERER> {
    ChargePos posBcast1[SCRATCH_PAD_WORK_GROUP_SIZE];
    uchar aboveThresholdBcast[SCRATCH_PAD_WORK_GROUP_SIZE];
    uchar buf[SCRATCH_PAD_WORK_GROUP_SIZE * SCRATCH_PAD_COUNT_N];
  };

  enum K : int {
    countPeaks,
  };

  static GPUd() void countPeaksImpl(int, int, int, int, GPUSharedMemory&, const Array2D<uchar>&, Array2D<PackedCharge>&, const deprecated::Digit*, const uint);

#ifdef HAVE_O2HEADERS
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

 private:
  static GPUd() char countPeaksAroundDigit(const ChargePos&, const Array2D<uchar>&);
  static GPUd() char countPeaksScratchpadInner(ushort, const uchar*, uchar*);
  static GPUd() char countPeaksScratchpadOuter(ushort, ushort, uchar, const uchar*);
};

} // namespace gpu
} // namespace GPUCA_NAMESPACE

#endif
