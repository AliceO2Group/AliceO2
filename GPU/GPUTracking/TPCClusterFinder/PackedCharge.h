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

#include "GPUDef.h"
#include "cl/clusterFinderDefs.h"

namespace GPUCA_NAMESPACE
{
namespace gpu
{

#if 0
class PackedCharge
{

 public:
  using BasicType = unsigned short;

  GPUd() PackedCharge(float);
  GPUd() PackedCharge(float, bool, bool);

  GPUd() float unpack() const;
  GPUd() bool has3x3Peak() const;
  GPUd() bool isSplit() const;

 private:
  GPUconstexpr() int ADCBits = 10;
  GPUconstexpr() int DecimalBits = 4;
  GPUconstexpr() int ChargeBits = ADCBits + DecimalBits;
  GPUconstexpr() int Has3x3PeakBit = ChargeBits;
  GPUconstexpr() BasicType Has3x3PeakMask = 1 << Has3x3PeakBit;
  GPUconstexpr() int IsSplitBit = ChargeBits + 1;
  GPUconstexpr() BasicType IsSplitMask = 1 << IsSplitBit;

  GPUconstexpr() BasicType ChargeMask = (1 << ChargeBits) - 1;
  GPUconstexpr() BasicType MaxVal = ChargeMask;
  GPUconstexpr() float Shift = 1 << DecimalBits;

  BasicType val;
};
#endif

using PackedCharge = ushort;

GPUd() PackedCharge packCharge(Charge, bool, bool);
GPUd() Charge unpackCharge(PackedCharge);
GPUd() bool has3x3Peak(PackedCharge);
GPUd() bool wasSplit(PackedCharge);

} // namespace gpu
} // namespace GPUCA_NAMESPACE

#endif
