// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file ChargeMapFiller.cxx
/// \author Felix Weiglhofer

#include "ChargeMapFiller.h"
#include "Array2D.h"

using namespace GPUCA_NAMESPACE::gpu;
using namespace GPUCA_NAMESPACE::gpu::deprecated;

GPUd() void ChargeMapFiller::fillChargeMapImpl(int nBlocks, int nThreads, int iBlock, int iThread, GPUTPCClusterFinderKernels::GPUTPCSharedMemory& smem,
                                               GPUglobalref() const Digit* digits,
                                               GPUglobalref() PackedCharge* chargeMap,
                                               size_t maxDigit)
{
  size_t idx = get_global_id(0);
  if (idx >= maxDigit) {
    return;
  }
  Digit myDigit = digits[idx];

  GlobalPad gpad = Array2D::tpcGlobalPadIdx(myDigit.row, myDigit.pad);

  CHARGE(chargeMap, gpad, myDigit.time) = PackedCharge(myDigit.charge, false, false);
}

GPUd() void ChargeMapFiller::resetMapsImpl(int nBlocks, int nThreads, int iBlock, int iThread, GPUTPCClusterFinderKernels::GPUTPCSharedMemory& smem,
                                           GPUglobalref() const Digit* digits,
                                           GPUglobalref() PackedCharge* chargeMap,
                                           GPUglobalref() uchar* isPeakMap)
{
  size_t idx = get_global_id(0);
  Digit myDigit = digits[idx];

  GlobalPad gpad = Array2D::tpcGlobalPadIdx(myDigit.row, myDigit.pad);

  CHARGE(chargeMap, gpad, myDigit.time) = PackedCharge(0);
  IS_PEAK(isPeakMap, gpad, myDigit.time) = 0;
}
