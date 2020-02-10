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
#include "GPUTPCClusterFinderKernels.h"
#include "Array2D.h"
#include "PackedCharge.h"

namespace GPUCA_NAMESPACE
{
namespace gpu
{

class Deconvolution : public GPUKernelTemplate
{

 public:
  class GPUTPCSharedMemory : public GPUKernelTemplate::GPUTPCSharedMemoryScan64<int, GPUCA_THREAD_COUNT_CLUSTERER>
  {
   public:
    GPUTPCSharedMemoryData::count_t count;
  };

  enum K : int {
    countPeaks = 5,
  };

  static GPUd() void countPeaksImpl(int, int, int, int, GPUTPCSharedMemory&, const Array2D<uchar>&, Array2D<PackedCharge>&, const deprecated::Digit*, const uint);

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

  template <int iKernel = 0, typename... Args>
  GPUd() static void Thread(int nBlocks, int nThreads, int iBlock, int iThread, GPUTPCSharedMemory& smem, processorType& clusterer, Args... args);

 private:
  static GPUd() char countPeaksAroundDigit(const ChargePos&, const Array2D<uchar>&);
  static GPUd() char countPeaksScratchpadInner(ushort, const uchar*, uchar*);
  static GPUd() char countPeaksScratchpadOuter(ushort, ushort, uchar, const uchar*);
};

} // namespace gpu
} // namespace GPUCA_NAMESPACE

#endif
