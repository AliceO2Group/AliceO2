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
#include "GPUTPCClusterFinderKernels.h"
#include "Array2D.h"
#include "PackedCharge.h"

namespace GPUCA_NAMESPACE
{
namespace gpu
{

class NoiseSuppression : GPUKernelTemplate
{

 public:
  class GPUTPCSharedMemory
  {
   public:
    GPUTPCSharedMemoryData::noise_t noise;
  };

  enum K : int {
    noiseSuppression,
    updatePeaks,
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

  template <int iKernel = 0, typename... Args>
  GPUd() static void Thread(int nBlocks, int nThreads, int iBlock, int iThread, GPUTPCSharedMemory& smem, processorType& clusterer, Args... args);

  static GPUd() void noiseSuppressionImpl(int, int, int, int, GPUTPCSharedMemory&, const Array2D<PackedCharge>&, const Array2D<uchar>&, const deprecated::Digit*, const uint, uchar*);

  static GPUd() void updatePeaksImpl(int, int, int, int, const deprecated::Digit*, const uchar*, Array2D<uchar>&);

 private:
  static GPUd() void checkForMinima(float, float, PackedCharge, int, ulong*, ulong*);

  static GPUd() void findMinimaScratchPad(const PackedCharge*, const ushort, const int, int, const float, const float, ulong*, ulong*);

  static GPUd() void findPeaksScratchPad(const uchar*, const ushort, const int, int, ulong*);

  static GPUd() void findMinima(const Array2D<PackedCharge>&, const ChargePos&, const float, const float, ulong*, ulong*);

  static GPUd() ulong findPeaks(const Array2D<uchar>&, const ChargePos&);

  static GPUd() bool keepPeak(ulong, ulong);

  static GPUd() void findMinimaAndPeaksScratchpad(const Array2D<PackedCharge>&, const Array2D<uchar>&, float, const ChargePos&, ChargePos*, PackedCharge*, ulong*, ulong*, ulong*);
};

} // namespace gpu
} // namespace GPUCA_NAMESPACE

#endif
