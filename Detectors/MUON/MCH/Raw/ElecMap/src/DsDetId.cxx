// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "MCHRawElecMap/DsDetId.h"
#include "Assertions.h"
#include <fmt/format.h>
#include <iostream>

namespace o2::mch::raw
{

DsDetId::DsDetId(int deId, int dsId)
{
  impl::assertIsInRange("deId", deId, 0, 1025);
  impl::assertIsInRange("dsId", dsId, 0, 1361);
  mDeId = static_cast<uint16_t>(deId);
  mDsId = static_cast<uint16_t>(dsId);
}

uint32_t encode(const DsDetId& id)
{
  return id.deId() << 16 | id.dsId();
}

DsDetId decodeDsDetId(uint32_t x)
{
  uint16_t deId = static_cast<uint16_t>((x & 0xFFFF0000) >> 16);
  uint16_t dsId = static_cast<uint16_t>(x & 0xFFFF);
  return DsDetId(deId, dsId);
}

std::ostream& operator<<(std::ostream& os, const DsDetId& id)
{
  os << fmt::format("DsDetId(DE={:4d},DS={:4d}) CODE={:8d}", id.deId(), id.dsId(), encode(id));
  return os;
}

std::string asString(DsDetId dsDetId)
{
  return fmt::format("DE{}-DS{}", dsDetId.deId(), dsDetId.dsId());
}

} // namespace o2::mch::raw
