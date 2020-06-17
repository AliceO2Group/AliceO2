// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "MapCRU.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include <algorithm>
#include <limits>

namespace
{
bool isValid(uint16_t solarId)
{
  return solarId != std::numeric_limits<uint16_t>::max();
}
} // namespace

namespace o2::mch::raw
{

MapCRU::MapCRU(std::string_view content)
{
  mFeeLink2Solar.fill(std::numeric_limits<uint16_t>::max());
  std::istringstream input(content.data());
  std::string s;
  while (std::getline(input, s)) {
    if (s.empty()) {
      continue;
    }
    std::istringstream line(s);
    int f, l, link_id;
    line >> link_id >> f >> l;
    auto ix = indexFeeLink(f, l);
    if (ix < 0) {
      continue;
    }
    mFeeLink2Solar.at(ix) = link_id;
  }
}

int MapCRU::indexFeeLink(int feeid, int linkid) const
{
  if (feeid < 0 || feeid >= sMaxFeeId) {
    return -1;
  }
  if (linkid < 0 || linkid >= sMaxLinkId) {
    return -1;
  }
  return feeid * sMaxLinkId + linkid;
}

size_t MapCRU::size() const
{
  return std::count_if(mFeeLink2Solar.begin(), mFeeLink2Solar.end(), [](uint16_t a) { return a != std::numeric_limits<uint16_t>::max(); });
}

std::optional<uint16_t> MapCRU::operator()(const o2::mch::raw::FeeLinkId& feeLinkId) const
{
  if (!size()) {
    return std::nullopt;
  }
  auto ix = indexFeeLink(feeLinkId.feeId(), feeLinkId.linkId());
  if (ix < 0) {
    return std::nullopt;
  }
  auto solarId = mFeeLink2Solar.at(ix);
  if (isValid(solarId)) {
    return solarId;
  }
  return std::nullopt;
}
} // namespace o2::mch::raw
