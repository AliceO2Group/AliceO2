// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "MCHRawElecMap/CruLinkId.h"
#include "Assertions.h"
#include <fmt/format.h>
#include <iostream>

namespace o2::mch::raw
{

CruLinkId::CruLinkId(uint16_t cruId, uint8_t linkId)
  : mCruId(cruId), mLinkId(linkId)
{
  impl::assertIsInRange("linkId", mLinkId, 0, 23);
}

CruLinkId::CruLinkId(uint16_t cruId, uint8_t linkId, uint16_t deId)
  : mCruId(cruId), mLinkId(linkId), mDeId(deId)
{
  impl::assertIsInRange("linkId", mLinkId, 0, 23);
}

uint32_t encode(const CruLinkId& id)
{
  return (id.deId() << 17) | (id.cruId() << 5) | id.linkId();
}

CruLinkId decodeCruLinkId(uint32_t x)
{
  uint16_t cruId = static_cast<uint16_t>((x & 0x3FFE0) >> 5);
  uint16_t linkId = static_cast<uint8_t>(x & 0x1F);
  uint16_t deId = static_cast<uint16_t>((x & 0xFFFC0000) >> 17);
  return CruLinkId(cruId, linkId, deId);
}

std::ostream& operator<<(std::ostream& os, const CruLinkId& id)
{
  os << fmt::format("CruLinkId(CRU={:4d},LINK={:1d},DE={:4d}) CODE={:8d}", id.cruId(), id.linkId(), id.deId(), encode(id));
  return os;
}

} // namespace o2::mch::raw
