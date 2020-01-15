// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include <map>
#include <cstdint>
#include "MCHRawElecMap/DsElecId.h"
#include "MCHRawElecMap/DsDetId.h"
#include "MCHRawElecMap/CruLinkId.h"
#include <stdexcept>
#include <fmt/format.h>
#include <iostream>
#include <algorithm>

namespace
{
void add(std::map<uint16_t, uint32_t>& e2d, int deId, int dsId,
         uint16_t solarId, uint8_t groupId, uint8_t index)
{
  e2d.emplace(encode(o2::mch::raw::DsElecId(solarId, groupId, index)), encode(o2::mch::raw::DsDetId(deId, dsId)));
}
void add_cru(std::map<uint16_t, uint32_t>& s2c, int cruId, int linkId, uint16_t solarId, uint16_t deId)
{
  auto code = encode(o2::mch::raw::CruLinkId(cruId, linkId, deId));
  // ensure we don't have duplicated codes in the map

  // std::cout << fmt::format("CRU {:4d} LINK {:2d} SOLAR {:4d} DE {:4d} CODE {:8d}\n",
  //                          cruId, linkId, solarId, deId, code);

  if (std::find_if(begin(s2c), end(s2c), [code](const auto& v) { return v.second == code; }) != end(s2c)) {
    throw std::logic_error(fmt::format("Seems cru,link,deId=({},{},{}) is already referenced in the map !",
                                       cruId, linkId, deId));
  }
  s2c.emplace(solarId, code);
}
} // namespace
