// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "MCHRawCommon/DataFormats.h"
#include <fmt/format.h>
#include <iostream>

namespace o2::mch::raw
{

template <>
uint16_t extraFeeIdChargeSumMask<ChargeSumMode>()
{
  FEEID f{0};
  f.chargeSum = 1;
  return f.word;
}

template <>
uint16_t extraFeeIdChargeSumMask<SampleMode>()
{
  return 0;
}
template <int VERSION>
uint16_t extraFeeIdVersionMask()
{
  FEEID f{0};
  f.ulFormatVersion = VERSION;
  return f.word;
}

template uint16_t extraFeeIdVersionMask<0>();
template uint16_t extraFeeIdVersionMask<1>();

template <>
uint8_t linkRemapping<UserLogicFormat>(uint8_t linkID)
{
  return 15;
}

template <>
uint8_t linkRemapping<BareFormat>(uint8_t linkID)
{
  return linkID;
}

std::ostream& operator<<(std::ostream& os, const FEEID& f)
{
  os << fmt::format("FEEID {} [id:{},chargeSum:{} ulVersion:{}]",
                    f.word, f.id, f.chargeSum, f.ulFormatVersion);
  return os;
}
} // namespace o2::mch::raw
