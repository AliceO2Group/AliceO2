// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file PeakFinder.h
/// \author Felix Weiglhofer

#ifndef O2_GPU_PEAK_FINDER_H
#define O2_GPU_PEAK_FINDER_H

#include "GPUGeneralKernels.h"
#include "GPUConstantMem.h"

#include "clusterFinderDefs.h"
#include "Array2D.h"
#include "PackedCharge.h"

namespace GPUCA_NAMESPACE::gpu
{

struct ChargePos;

class GPUTPCCFPeakFinder : public GPUKernelTemplate
{
 public:
  static constexpr size_t SCRATCH_PAD_WORK_GROUP_SIZE = GPUCA_GET_THREAD_COUNT(GPUCA_LB_GPUTPCCFPeakFinder);
  struct GPUSharedMemory : public GPUKernelTemplate::GPUSharedMemoryScan64<short, GPUCA_GET_THREAD_COUNT(GPUCA_LB_GPUTPCCFPeakFinder)> {
    ChargePos posBcast[SCRATCH_PAD_WORK_GROUP_SIZE];
    PackedCharge buf[SCRATCH_PAD_WORK_GROUP_SIZE * SCRATCH_PAD_SEARCH_N];
  };

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
  static GPUd() void findPeaksImpl(int, int, int, int, GPUSharedMemory&, const Array2D<PackedCharge>&, const uchar*, const ChargePos*, tpccf::SizeT, const GPUSettingsRec&, const TPCPadGainCalib&, uchar*, Array2D<uchar>&);

  static GPUd() bool isPeak(GPUSharedMemory&, tpccf::Charge, const ChargePos&, ushort, const Array2D<PackedCharge>&, const GPUSettingsRec&, ChargePos*, PackedCharge*);
};

} // namespace GPUCA_NAMESPACE::gpu

#endif
