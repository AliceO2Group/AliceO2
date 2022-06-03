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

#ifndef O2_MCH_WORKFLOW_MAPCRU_H
#define O2_MCH_WORKFLOW_MAPCRU_H

#include <string_view>
#include <array>
#include <cstdlib>
#include <optional>
#include "MCHRawElecMap/FeeLinkId.h"
#include <set>

namespace o2::mch::raw
{
class MapCRU
{
 public:
  MapCRU(std::string_view content);
  std::optional<uint16_t> operator()(const FeeLinkId& feeLinkId) const;
  size_t size() const;
  std::set<uint16_t> getSolarUIDs() const;
  std::optional<FeeLinkId> operator()(uint16_t solarId) const;

 private:
  int indexFeeLink(int feeid, int linkid) const;

 private:
  static constexpr int sMaxFeeId = 64;
  static constexpr int sMaxLinkId = 12;
  std::array<uint16_t, sMaxFeeId * sMaxLinkId> mFeeLink2Solar;
  size_t mSize = 0;
};

} // namespace o2::mch::raw
#endif
