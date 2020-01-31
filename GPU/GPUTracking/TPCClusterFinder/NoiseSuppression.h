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

class NoiseSuppression
{

 public:
  static GPUd() void noiseSuppressionImpl(int, int, int, int, GPUTPCClusterFinderKernels::GPUTPCSharedMemory&, const Array2D<PackedCharge>&, const Array2D<uchar>&, GPUglobalref() const deprecated::Digit*, const uint, GPUglobalref() uchar*);

  static GPUd() void updatePeaksImpl(int, int, int, int, GPUTPCClusterFinderKernels::GPUTPCSharedMemory&, GPUglobalref() const deprecated::Digit*, GPUglobalref() const uchar*, Array2D<uchar>&);

 private:
  static GPUd() void checkForMinima(float, float, PackedCharge, int, ulong*, ulong*);

  static GPUd() void findMinimaScratchPad(GPUsharedref() const PackedCharge*, const ushort, const int, int, const float, const float, ulong*, ulong*);

  static GPUd() void findPeaksScratchPad(GPUsharedref() const uchar*, const ushort, const int, int, ulong*);

  static GPUd() void findMinima(const Array2D<PackedCharge>&, const ChargePos&, const float, const float, ulong*, ulong*);

  static GPUd() ulong findPeaks(const Array2D<uchar>&, const ChargePos&, bool);

  static GPUd() bool keepPeak(ulong, ulong);

  static GPUd() void findMinimaAndPeaksScratchpad(const Array2D<PackedCharge>&, const Array2D<uchar>&, float, const ChargePos&, GPUsharedref() ChargePos*, GPUsharedref() PackedCharge*, ulong*, ulong*, ulong*);
};

} // namespace gpu
} // namespace GPUCA_NAMESPACE

#endif
