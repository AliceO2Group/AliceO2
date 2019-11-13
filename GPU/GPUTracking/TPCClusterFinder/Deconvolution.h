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
#include "PackedCharge.h"

namespace GPUCA_NAMESPACE
{
namespace gpu
{

class Deconvolution
{

 public:
  static GPUd() void countPeaksImpl(int, int, int, int, GPUTPCClusterFinderKernels::GPUTPCSharedMemory&, GPUglobalref() const uchar*, GPUglobalref() PackedCharge*, GPUglobalref() const deprecated::Digit*, const uint);

 private:
  static GPUd() char countPeaksAroundDigit(const GlobalPad, const Timestamp, GPUglobalref() const uchar*);
  static GPUd() char countPeaksScratchpadInner(ushort, GPUsharedref() const uchar*, uchar*);
  static GPUd() char countPeaksScratchpadOuter(ushort, ushort, uchar, GPUsharedref() const uchar*);
};

} // namespace gpu
} // namespace GPUCA_NAMESPACE

#endif
