// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file NoiseSuppression.h
/// \author Felix Weiglhofer

#ifndef O2_GPU_NOISE_SUPPRESSION_H
#define O2_GPU_NOISE_SUPPRESSION_H

#include "clusterFinderDefs.h"
#include "GPUGeneralKernels.h"
#include "GPUConstantMem.h"
#include "GPUTPCClusterFinder.h"
#include "Array2D.h"
#include "PackedCharge.h"

namespace GPUCA_NAMESPACE::gpu
{

struct ChargePos;

class GPUTPCCFNoiseSuppression : public GPUKernelTemplate
{
 public:
  enum K : int {
    noiseSuppression = 0,
    updatePeaks = 1,
  };
  static constexpr size_t SCRATCH_PAD_WORK_GROUP_SIZE = GPUCA_GET_THREAD_COUNT(GPUCA_LB_GPUTPCCFNoiseSuppression);
  struct GPUSharedMemory {
    ChargePos posBcast[SCRATCH_PAD_WORK_GROUP_SIZE];
    PackedCharge buf[SCRATCH_PAD_WORK_GROUP_SIZE * SCRATCH_PAD_NOISE_N];
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
  static GPUd() void noiseSuppressionImpl(int, int, int, int, GPUSharedMemory&, const GPUSettingsRec&, const Array2D<PackedCharge>&, const Array2D<uchar>&, const ChargePos*, const uint, uchar*);

  static GPUd() void updatePeaksImpl(int, int, int, int, const ChargePos*, const uchar*, const uint, Array2D<uchar>&);

  static GPUdi() void checkForMinima(float, float, PackedCharge, int, ulong*, ulong*);

  static GPUdi() void findMinima(const PackedCharge*, const ushort, const int, int, const float, const float, ulong*, ulong*);

  static GPUdi() void findPeaks(const uchar*, const ushort, const int, int, ulong*);

  static GPUdi() bool keepPeak(ulong, ulong);

  static GPUd() void findMinimaAndPeaks(const Array2D<PackedCharge>&, const Array2D<uchar>&, const GPUSettingsRec&, float, const ChargePos&, ChargePos*, PackedCharge*, ulong*, ulong*, ulong*);
};

} // namespace GPUCA_NAMESPACE::gpu

#endif
