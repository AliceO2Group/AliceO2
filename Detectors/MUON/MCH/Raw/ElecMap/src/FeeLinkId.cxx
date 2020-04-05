// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "MCHRawElecMap/FeeLinkId.h"
#include <fmt/format.h>
#include <iostream>

namespace o2::mch::raw
{
FeeLinkId::FeeLinkId(uint16_t feeId, uint8_t linkId) : mFeeId{feeId},
                                                       mLinkId{linkId}
{
  if (mLinkId < 0 || (mLinkId > 11 && mLinkId != 15)) {
    throw std::invalid_argument(fmt::format("LinkId should be between 0 and 11 or = 15 but is {}", linkId));
  }
}
uint32_t encode(const FeeLinkId& id)
{
  return (id.feeId() << 16) | id.linkId();
}

FeeLinkId decodeFeeLinkId(uint32_t x)
{
  uint16_t feeId = static_cast<uint16_t>((x & 0xFFFF0000) >> 16);
  uint16_t linkId = static_cast<uint8_t>(x & 0xF);
  return FeeLinkId(feeId, linkId);
}

std::ostream& operator<<(std::ostream& os, const FeeLinkId& id)
{
  os << fmt::format("FeeLinkId(FEE={:4d},LINK={:1d}) CODE={:8d}", id.feeId(), id.linkId(), encode(id));
  return os;
}

std::string asString(const FeeLinkId& feeLinkId)
{
  return fmt::format("FEE{}-LINK{}", feeLinkId.feeId(), feeLinkId.linkId());
}

bool operator<(const FeeLinkId& f1, const FeeLinkId& f2)
{
  return encode(f1) < encode(f2);
}
} // namespace o2::mch::raw
