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

#ifndef O2_MCH_RAW_ELECMAP_MAP_FEC_H
#define O2_MCH_RAW_ELECMAP_MAP_FEC_H

#include <array>
#include <istream>
#include "MCHRawElecMap/DsDetId.h"
#include "MCHRawElecMap/DsElecId.h"
#include <optional>

namespace o2::mch::raw
{
class MapFEC
{
 public:
  MapFEC(std::string_view content);
  std::optional<DsDetId> operator()(const DsElecId& elecId) const;
  size_t size() const;

 private:
  int index(uint32_t linkId, uint32_t dsAddr) const;

 private:
  struct MapDualSampa {
    int deId = -1; // detector element
    int dsId = -1; // DS index
    int bad = -1;  // if = 1 bad pad (not used for analysis)
  };

 private:
  static constexpr int sMaxLinkId = 0x7ff;
  static constexpr int sMaxDs = 40;
  std::array<MapDualSampa, (sMaxLinkId + 1) * sMaxDs> mDsMap;
  size_t mSize = 0;
};

} // namespace o2::mch::raw

#endif
