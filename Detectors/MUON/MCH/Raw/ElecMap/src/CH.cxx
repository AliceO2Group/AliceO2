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
#include "MCHRawElecMap/FeeLinkId.h"
#include <stdexcept>
#include <fmt/format.h>
#include <iostream>
#include <algorithm>

namespace
{
void add(std::map<uint32_t, uint32_t>& e2d, int deId, int dsId,
         uint16_t solarId, uint8_t groupId, uint8_t index)
{
  o2::mch::raw::DsElecId dselec(solarId, groupId, index);
  auto code = encode(dselec);
  auto already = e2d.find(code);

  // ensure we don't try to use already used key in the map
  if (already != e2d.end()) {
    auto previous = o2::mch::raw::decodeDsDetId(already->second);
    throw std::logic_error(fmt::format("FATAL_ERROR: dselec={} (deId,dsId={},{}) is already in the map for (deId,dsId={})",
                                       o2::mch::raw::asString(dselec), deId, dsId, o2::mch::raw::asString(previous)));
  }

  // // ensure we don't have duplicated codes in the map
  // if (std::find_if(begin(e2d), end(e2d), [code](const auto& v) { return v.second == code; }) != end(e2d)) {
  //   throw std::logic_error(fmt::format("FATAL_ERROR dselec={} (de,ds={},{}) is already referenced in the map !",
  //                                      o2::mch::raw::asString(dselec), deId, dsId));
  // }
  e2d.emplace(code, encode(o2::mch::raw::DsDetId(deId, dsId)));
}

void add_cru(std::map<uint16_t, uint32_t>& s2f, int feeId, int linkId, uint16_t solarId)
{
  auto code = encode(o2::mch::raw::FeeLinkId(feeId, linkId));
  // ensure we don't have duplicated codes in the map
  if (std::find_if(begin(s2f), end(s2f), [code](const auto& v) { return v.second == code; }) != end(s2f)) {
    throw std::logic_error(fmt::format("FATAL_ERROR feeid,link=({},{}) (solarId={}) is already referenced in the map !",
                                       feeId, linkId, solarId));
  }
  s2f.emplace(solarId, code);
}
} // namespace
