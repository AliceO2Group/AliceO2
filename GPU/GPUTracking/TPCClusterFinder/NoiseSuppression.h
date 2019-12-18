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
#include "PackedCharge.h"

namespace GPUCA_NAMESPACE
{
namespace gpu
{

class NoiseSuppression
{

 public:
  static GPUd() void noiseSuppressionImpl(int, int, int, int, GPUTPCClusterFinderKernels::GPUTPCSharedMemory&, GPUglobalref() const PackedCharge*, GPUglobalref() const uchar*, GPUglobalref() const deprecated::Digit*, const uint, GPUglobalref() uchar*);

  static GPUd() void updatePeaksImpl(int, int, int, int, GPUTPCClusterFinderKernels::GPUTPCSharedMemory&, GPUglobalref() const deprecated::Digit*, GPUglobalref() const uchar*, GPUglobalref() uchar*);

 private:
  static GPUd() void checkForMinima(float, float, PackedCharge, int, ulong*, ulong*);

  static GPUd() void findMinimaScratchPad(GPUsharedref() const PackedCharge*, const ushort, const int, int, const float, const float, ulong*, ulong*);

  static GPUd() void findPeaksScratchPad(GPUsharedref() const uchar*, const ushort, const int, int, ulong*);

  static GPUd() void findMinima(GPUglobalref() const PackedCharge*, const GlobalPad, const Timestamp, const float, const float, ulong*, ulong*);

  static GPUd() ulong findPeaks(GPUglobalref() const uchar*, const GlobalPad, const Timestamp, bool);

  static GPUd() bool keepPeak(ulong, ulong);

  static GPUd() void findMinimaAndPeaksScratchpad(GPUglobalref() const PackedCharge*, GPUglobalref() const uchar*, float, GlobalPad, Timestamp, GPUsharedref() ChargePos*, GPUsharedref() PackedCharge*, ulong*, ulong*, ulong*);
};

} // namespace gpu
} // namespace GPUCA_NAMESPACE

#endif
