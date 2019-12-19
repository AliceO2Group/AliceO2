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

#ifndef O2_GPU_PACKED_CHARGE_H
#define O2_GPU_PACKED_CHARGE_H

#include "clusterFinderDefs.h"
#include "GPUCommonMath.h"

namespace GPUCA_NAMESPACE
{
namespace gpu
{

// Ugly workaround because cuda doesn't like member constants
namespace PackedChargeDefs
{
using BasicType = unsigned short;
GPUconstexpr() int ADCBits = 10;
GPUconstexpr() int DecimalBits = 4;
GPUconstexpr() int ChargeBits = 14;
GPUconstexpr() int Has3x3PeakBit = 14;
GPUconstexpr() BasicType Has3x3PeakMask = 1 << 14;
GPUconstexpr() int IsSplitBit = 15;
GPUconstexpr() BasicType IsSplitMask = 1 << 15;

GPUconstexpr() BasicType ChargeMask = (1 << 14) - 1;
GPUconstexpr() BasicType MaxVal = (1 << 14) - 1;
GPUconstexpr() Charge Shift = 16.f;
} // namespace PackedChargeDefs

class PackedCharge
{
 public:
  using BasicType = PackedChargeDefs::BasicType;

  PackedCharge() = default;
  GPUdi() explicit PackedCharge(Charge q) : PackedCharge(q, false, false) {}
  GPUdi() PackedCharge(Charge q, bool peak3x3, bool wasSplit)
  {
    val = q * PackedChargeDefs::Shift;
    val = CAMath::Min(PackedChargeDefs::MaxVal, val); // ensure only lower 14 bits are set
    val |= (BasicType(peak3x3) << PackedChargeDefs::Has3x3PeakBit);
    val |= (BasicType(wasSplit) << PackedChargeDefs::IsSplitBit);
  }

  GPUdi() Charge unpack() const { return Charge(val & PackedChargeDefs::ChargeMask) / PackedChargeDefs::Shift; }
  GPUdi() bool has3x3Peak() const { return val & PackedChargeDefs::Has3x3PeakMask; }
  GPUdi() bool isSplit() const { return val & PackedChargeDefs::IsSplitMask; }

 private:
  BasicType val;
};

} // namespace gpu
} // namespace GPUCA_NAMESPACE

#endif
