// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file PackedCharge.h
/// \author Felix Weiglhofer

#include "PackedCharge.h"

#include "GPUCommonMath.h"

using namespace GPUCA_NAMESPACE::gpu;

GPUd() PackedCharge::PackedCharge(float q) : PackedCharge(q, false, false) {}

GPUd() PackedCharge::PackedCharge(float q, bool isSplit, bool has3x3Peak)
{
  val = q * Shift;
  val = CAMath::Min(MaxVal, val);
  val |= (has3x3Peak << Has3x3PeakBit);
  val |= (isSplit << IsSplitBit);
}

GPUd() float PackedCharge::unpack() const
{
  return (val & ChargeMask) / Shift;
}

GPUd() bool PackedCharge::has3x3Peak() const
{
  return val & Has3x3PeakMask;
}

GPUd() bool PackedCharge::isSplit() const
{
  return val & IsSplitMask;
}
