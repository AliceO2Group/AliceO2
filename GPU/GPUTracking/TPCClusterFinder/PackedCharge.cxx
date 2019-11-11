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

#if 0
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
#endif

namespace GPUCA_NAMESPACE
{
namespace gpu
{

GPUd() PackedCharge packCharge(Charge q, bool peak3x3, bool wasSplit)
{
  PackedCharge p = q * 16.f;
  p = CAMath::Min((PackedCharge)0x3FFF, p); // ensure only lower 14 bits are set
  p |= (wasSplit << 14);
  p |= (peak3x3 << 15);
  return p;
}

GPUd() Charge unpackCharge(PackedCharge p)
{
  return (p & 0x3FFF) / 16.f;
}

GPUd() bool has3x3Peak(PackedCharge p)
{
  return p & (1 << 15);
}

GPUd() bool wasSplit(PackedCharge p)
{
  return p & (1 << 14);
}

} // namespace gpu
} // namespace GPUCA_NAMESPACE
