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
#include "MCHRawElecMap/Mapper.h"
#include "dslist.h"
#include "DetectionElements.h"

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

template <typename T>
std::set<uint16_t> getSolarUIDs(int deid)
{
  static auto d2e = o2::mch::raw::createDet2ElecMapper<T>();
  std::set<uint16_t> solarsForDE;
  static auto dslist = createDualSampaMapper();
  for (auto dsid : dslist(deid)) {
    DsDetId id{static_cast<uint16_t>(deid), static_cast<uint16_t>(dsid)};
    auto dsel = d2e(id);
    if (dsel.has_value()) {
      solarsForDE.insert(dsel->solarId());
    }
  }
  return solarsForDE;
}

template <typename T>
std::set<uint16_t> getSolarUIDs()
{
  std::set<uint16_t> solarUIDs;

  for (auto deid : o2::mch::constants::deIdsForAllMCH) {
    std::set<uint16_t> solarsForDE = getSolarUIDs<T>(deid);
    for (auto s : solarsForDE) {
      solarUIDs.insert(s);
    }
  }
  return solarUIDs;
}

template <typename T>
std::set<uint16_t> getSolarUIDsPerFeeId(uint16_t feeid)
{
  std::set<uint16_t> solars;
  static auto feeLink2Solar = createFeeLink2SolarMapper<T>();
  for (uint8_t link = 0; link < 12; link++) {
    auto solar = feeLink2Solar(FeeLinkId{feeid, link});
    if (solar.has_value()) {
      solars.insert(solar.value());
    }
  }
  return solars;
}

template <typename T>
std::set<DsDetId> getDualSampasPerFeeId(uint16_t feeId)
{
  auto solarIds = getSolarUIDsPerFeeId<T>(feeId);
  int n{0};
  std::set<DsDetId> allDualSampas;
  for (auto solarId : solarIds) {
    auto solarDualSampas = getDualSampas<T>(solarId);
    allDualSampas.insert(solarDualSampas.begin(), solarDualSampas.end());
  }
  return allDualSampas;
}

template <typename T>
std::vector<std::string> solar2FeeLinkConsistencyCheck()
{
  std::vector<std::string> errors;

  // All solars must have a FeeLinkId
  std::set<uint16_t> solarIds = getSolarUIDs<T>();
  static auto solar2feeLink = createSolar2FeeLinkMapper<T>();
  std::vector<o2::mch::raw::FeeLinkId> feeLinkIds;
  for (auto s : solarIds) {
    auto p = solar2feeLink(s);
    if (!p.has_value()) {
      errors.push_back(fmt::format("Got no feelinkId for solarId {}", s));
    } else {
      feeLinkIds.push_back(p.value());
    }
  }

  // All FeeLinkId must have a SolarId
  auto feeLinkId2SolarId = createFeeLink2SolarMapper<T>();
  for (auto f : feeLinkIds) {
    auto p = feeLinkId2SolarId(f);
    if (!p.has_value()) {
      errors.push_back(fmt::format("Got no solarId for FeeLinkId {}", asString(f)));
    }
  }
  return errors;
}

template <typename T>
std::set<DsElecId> getAllDs()
{
  std::set<DsElecId> dsElecIds;

  static auto dslist = createDualSampaMapper();
  static auto det2ElecMapper = createDet2ElecMapper<T>();

  for (auto deId : o2::mch::constants::deIdsForAllMCH) {
    for (auto dsId : dslist(deId)) {
      auto dsElecId = det2ElecMapper(DsDetId{deId, dsId});
      if (dsElecId.has_value()) {
        dsElecIds.insert(dsElecId.value());
      }
    }
  }

  return dsElecIds;
}

template <typename T>
std::set<DsDetId> getDualSampas(uint16_t solarId)
{
  std::set<DsDetId> dualSampas;

  static auto elec2det = o2::mch::raw::createElec2DetMapper<T>();
  for (uint8_t group = 0; group < 8; group++) {
    for (uint8_t index = 0; index < 5; index++) {
      DsElecId dsElecId{solarId, group, index};
      auto dsDetId = elec2det(dsElecId);
      if (dsDetId.has_value()) {
        dualSampas.insert(dsDetId.value());
      }
    }
  }
  return dualSampas;
}

} // namespace o2::mch::raw::impl

#endif
