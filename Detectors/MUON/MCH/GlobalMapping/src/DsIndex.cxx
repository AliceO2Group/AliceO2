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

#include "MCHGlobalMapping/DsIndex.h"

#include <map>
#include <stdexcept>
#include <vector>

#include "Framework/Logger.h"
#include "MCHMappingInterface/Segmentation.h"

namespace o2::mch
{

template <typename KEY, typename VALUE>
std::map<VALUE, KEY> inverseMap(const std::map<KEY, VALUE>& src)
{
  std::map<VALUE, KEY> dest;
  for (auto p : src) {
    dest.emplace(p.second, p.first);
  }
  return dest;
}

uint8_t numberOfDualSampaChannels(DsIndex dsIndex)
{
  static std::vector<uint8_t> channelsPerDS;
  if (channelsPerDS.empty()) {
    for (uint16_t dsIndex = 0; dsIndex < o2::mch::NumberOfDualSampas; ++dsIndex) {
      raw::DsDetId det{o2::mch::getDsDetId(dsIndex)};
      uint8_t nch{0};
      auto dsId = det.dsId();
      auto deId = det.deId();
      const auto& seg = o2::mch::mapping::segmentation(deId);
      seg.forEachPadInDualSampa(dsId, [&nch](int /*dePadIndex*/) { ++nch; });
      channelsPerDS.emplace_back(nch);
    }
  }
  if (dsIndex < o2::mch::NumberOfDualSampas) {
    return channelsPerDS[dsIndex];
  } else {
    LOGP(error, "invalid Dual Sampa index: {}", dsIndex);
  }
  return 0;
}

std::map<uint32_t, uint16_t> buildDetId2DsIndexMap()
{
  static std::map<uint32_t, uint16_t> m;
  if (m.empty()) {
    uint16_t dsIndex{0};
    o2::mch::mapping::forEachDetectionElement([&](int deId) {
      const auto& seg = o2::mch::mapping::segmentation(deId);
      std::vector<int> dsids;
      seg.forEachDualSampa([&](int dsid) {
        dsids.emplace_back(dsid);
      });
      // ensure dual sampa are sorted by dsid
      std::sort(dsids.begin(), dsids.end());
      for (auto dsId : dsids) {
        raw::DsDetId det{deId, dsId};
        m.emplace(encode(det), dsIndex);
        ++dsIndex;
      }
    });
  }
  return m;
}

DsIndex getDsIndex(const o2::mch::raw::DsDetId& dsDetId)
{
  static std::map<uint32_t, uint16_t> m = buildDetId2DsIndexMap();
  try {
    return m.at(encode(dsDetId));
  } catch (const std::exception&) {
    LOGP(error, "invalid Dual Sampa Id: {}", raw::asString(dsDetId));
  }
  return NumberOfDualSampas;
}

o2::mch::raw::DsDetId getDsDetId(DsIndex dsIndex)
{
  static std::map<uint16_t, uint32_t> m = inverseMap(buildDetId2DsIndexMap());
  try {
    return raw::decodeDsDetId(m.at(dsIndex));
  } catch (const std::exception&) {
    LOGP(error, "invalid Dual Sampa index: {}", dsIndex);
  }
  return raw::DsDetId(0, 0);
}

} // namespace o2::mch
