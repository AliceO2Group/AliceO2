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

#include "clusterFinderDefs.h"
#include "GPUTPCClusterFinderKernels.h"
#include "Array2D.h"

namespace GPUCA_NAMESPACE
{
namespace gpu
{

class PeakFinder
{

 public:
  static GPUd() void findPeaksImpl(int, int, int, int, GPUTPCClusterFinderKernels::GPUTPCSharedMemory&, const Array2D<gpu::PackedCharge>&, const deprecated::Digit*, uint, uchar*, Array2D<uchar>&);

 private:
  static GPUd() bool isPeakScratchPad(GPUTPCClusterFinderKernels::GPUTPCSharedMemory&, Charge, const ChargePos&, ushort, const Array2D<o2::gpu::PackedCharge>&, ChargePos*, PackedCharge*);

  static GPUd() bool isPeak(Charge, const ChargePos&, const Array2D<PackedCharge>&);
};

} // namespace gpu
} // namespace GPUCA_NAMESPACE

#endif
