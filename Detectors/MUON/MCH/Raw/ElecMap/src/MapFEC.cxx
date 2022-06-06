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

#include "MapFEC.h"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <string>
#include <sstream>

namespace o2::mch::raw
{

MapFEC::MapFEC(std::string_view content)
{
  mDsMap.fill({-1, -1, -1});
  int link_id, group_id, de, ds_id[5];
  std::istringstream in(std::string{content});
  while (in >> link_id >> group_id >> de >> ds_id[0] >> ds_id[1] >> ds_id[2] >> ds_id[3] >> ds_id[4]) {
    for (int i = 0; i < 5; i++) {
      if (ds_id[i] <= 0) {
        continue;
      }
      int ds_addr = group_id * 5 + i;
      int ix = index(link_id, ds_addr);
      if (ix < 0) {
        continue;
      }
      mDsMap.at(ix) = {de, ds_id[i], 0};
    }
  }

  mSize = size();
}

int MapFEC::index(uint32_t linkId, uint32_t dsAddr) const
{
  if (linkId < 0 || linkId > sMaxLinkId) {
    return -1;
  }
  if (dsAddr < 0 || dsAddr >= sMaxDs) {
    return -1;
  }
  return linkId * sMaxDs + dsAddr;
}

size_t MapFEC::size() const
{
  return std::count_if(mDsMap.begin(), mDsMap.end(), [](const MapDualSampa& m) {
    return m.deId >= 0 && m.dsId >= 0 && m.bad == 0;
  });
}

std::optional<o2::mch::raw::DsDetId> MapFEC::operator()(const o2::mch::raw::DsElecId& dsElecId) const
{
  auto link_id = dsElecId.solarId();
  auto ds_addr = dsElecId.elinkId();

  if (!mSize) {
    return std::nullopt;
  }

  int ix = index(link_id, ds_addr);
  if (ix < 0) {
    return std::nullopt;
  }
  if (mDsMap.at(ix).bad == 1) {
    return std::nullopt;
  }
  auto de = mDsMap.at(ix).deId;
  auto dsid = mDsMap.at(ix).dsId;
  if (de >= 0 && dsid >= 0) {
    return o2::mch::raw::DsDetId(de, dsid);
  }
  return std::nullopt;
}
} // namespace o2::mch::raw
