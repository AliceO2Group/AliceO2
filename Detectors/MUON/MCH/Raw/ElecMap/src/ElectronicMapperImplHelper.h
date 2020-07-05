// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_MCH_RAW_ELECMAP_ELECTRONIC_MAPPER_IMPL_HELPER_H
#define O2_MCH_RAW_ELECMAP_ELECTRONIC_MAPPER_IMPL_HELPER_H

#include "MCHRawElecMap/DsDetId.h"
#include "MCHRawElecMap/DsElecId.h"
#include "MCHRawElecMap/FeeLinkId.h"
#include <functional>
#include <optional>
#include <map>
#include <cstdint>
#include <set>
#include <fmt/format.h>
#include <iostream>

namespace o2::mch::raw::impl
{
template <typename T>
std::function<std::optional<o2::mch::raw::DsDetId>(o2::mch::raw::DsElecId)>
  mapperElec2Det(const std::map<uint32_t, uint32_t>& elec2det)
{
  return [elec2det](o2::mch::raw::DsElecId id) -> std::optional<o2::mch::raw::DsDetId> {
    auto it = elec2det.find(encode(id));
    if (it == elec2det.end()) {
      return std::nullopt;
    }
    return o2::mch::raw::decodeDsDetId(it->second);
  };
}

template <typename T>
std::function<std::optional<o2::mch::raw::DsElecId>(o2::mch::raw::DsDetId)>
  mapperDet2Elec(const std::map<uint32_t, uint32_t>& det2elec)
{
  return [det2elec](o2::mch::raw::DsDetId id) -> std::optional<o2::mch::raw::DsElecId> {
    auto it = det2elec.find(encode(id));
    if (it == det2elec.end()) {
      return std::nullopt;
    }
    return o2::mch::raw::decodeDsElecId(it->second);
  };
}

template <typename T>
std::function<std::optional<FeeLinkId>(uint16_t)>
  mapperSolar2FeeLink(const std::map<uint16_t, uint32_t>& solar2cruLink)
{
  return [solar2cruLink](uint16_t solarId) -> std::optional<FeeLinkId> {
    auto it = solar2cruLink.find(solarId);
    if (it == solar2cruLink.end()) {
      return std::nullopt;
    }
    return decodeFeeLinkId(it->second);
  };
}

template <typename T>
std::function<std::optional<uint16_t>(FeeLinkId)>
  mapperFeeLink2Solar(const std::map<uint32_t, uint16_t>& cruLink2solar)
{
  return [cruLink2solar](o2::mch::raw::FeeLinkId id) -> std::optional<uint16_t> {
    auto it = cruLink2solar.find(encode(id));
    if (it == cruLink2solar.end()) {
      return std::nullopt;
    }
    return it->second;
  };
}

template <typename KEY, typename VALUE>
std::map<VALUE, KEY> inverseMap(const std::map<KEY, VALUE>& src)
{
  std::map<VALUE, KEY> dest;
  for (auto p : src) {
    dest.emplace(p.second, p.first);
  }
  return dest;
}
} // namespace o2::mch::raw::impl

#endif
