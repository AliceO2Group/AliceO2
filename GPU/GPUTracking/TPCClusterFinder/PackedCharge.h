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

namespace GPUCA_NAMESPACE::gpu
{

class PackedCharge
{
 public:
  using BasicType = unsigned short;
  static_assert(sizeof(BasicType) == 2);

  enum Constants : BasicType {
    Null = 0,
    ADCBits = 10,
    DecimalBits = 4,
    ChargeBits = ADCBits + DecimalBits,
    ChargeMask = (1 << ChargeBits) - 1,
    MaxVal = ChargeMask,
    Has3x3PeakMask = 1 << ChargeBits,
    IsSplitMask = 1 << (ChargeBits + 1),
  };

  GPUdDefault() PackedCharge() CON_DEFAULT;
  GPUdi() explicit PackedCharge(tpccf::Charge q) : PackedCharge(q, false, false) {}
  GPUdi() PackedCharge(tpccf::Charge q, bool peak3x3, bool wasSplit)
  {
    mVal = q * tpccf::Charge(1 << DecimalBits);
    mVal = CAMath::Min<BasicType>(MaxVal, mVal); // ensure only lower 14 bits are set
    mVal |= (peak3x3) ? Has3x3PeakMask : Null;
    mVal |= (wasSplit) ? IsSplitMask : Null;
  }

  GPUdi() tpccf::Charge unpack() const { return tpccf::Charge(mVal & ChargeMask) / tpccf::Charge(1 << DecimalBits); }
  GPUdi() bool has3x3Peak() const { return mVal & Has3x3PeakMask; }
  GPUdi() bool isSplit() const { return mVal & IsSplitMask; }
  GPUdi() bool isZero() const { return mVal == 0; }

 private:
  BasicType mVal;
};

} // namespace GPUCA_NAMESPACE::gpu

#endif
