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

#include "MCHRawElecMap/Mapper.h"
#include <map>
#include <fmt/format.h>
#include "MCHRawElecMap/DsElecId.h"
#include "MCHRawElecMap/DsDetId.h"
#include "MCHRawElecMap/FeeLinkId.h"
#include "ElectronicMapperImplHelper.h"
#include "dslist.h"

namespace
{

constexpr int firstSolarId{360}; // so at least we get some overlap with real solarIds

// build the map to go from electronic ds id to detector ds id
std::map<uint32_t, uint32_t> buildDsElecId2DsDetIdMap()
{
  std::map<uint32_t, uint32_t> e2d;

  auto dslist = createDualSampaMapper();

  uint16_t n{0};
  uint16_t solarId{firstSolarId};
  uint8_t groupId{0};
  uint8_t index{0};

  for (auto deId : o2::mch::constants::deIdsForAllMCH) {
    // assign a tuple (solarId,groupId,index) to the pair (deId,dsId)
    for (auto dsId : dslist(deId)) {
      // index 0..4
      // groupId 0..7
      // solarId 0..nsolars
      if (n % 5 == 0) {
        index = 0;
        if (n % 8 == 0) {
          groupId = 0;
        } else {
          groupId++;
        }
      } else {
        index++;
      }
      if (n % 40 == 0) {
        solarId++;
      }
      o2::mch::raw::DsElecId dsElecId(solarId, groupId, index);
      o2::mch::raw::DsDetId dsDetId(deId, dsId);
      e2d.emplace(o2::mch::raw::encode(dsElecId),
                  o2::mch::raw::encode(dsDetId));
      n++;
    };
  };
  return e2d;
}

std::map<uint32_t, uint16_t> buildFeeLinkId2SolarIdMap()
{
  std::map<uint32_t, uint16_t> c2s;

  uint16_t n{0};
  uint16_t solarId{firstSolarId};

  auto dslist = createDualSampaMapper();

  std::set<uint16_t> solarIds;

  for (auto deId : o2::mch::constants::deIdsForAllMCH) {
    // assign a tuple (fee,link) to each solarId
    for (auto dsId : dslist(deId)) {
      if (n % 40 == 0) {
        solarId++;
        solarIds.insert(solarId);
      }
      n++;
    }
  }
  for (auto solarId : solarIds) {
    auto feeId = solarId / 12;
    auto linkId = solarId - feeId * 12;
    c2s[encode(o2::mch::raw::FeeLinkId(feeId, linkId))] = solarId;
  }
  return c2s;
}
} // namespace

namespace o2::mch::raw
{

template <>
std::function<std::optional<DsDetId>(DsElecId)>
  createElec2DetMapper<ElectronicMapperDummy>(uint64_t timestamp)
{
  static std::map<uint32_t, uint32_t> dsElecId2DsDetId = buildDsElecId2DsDetIdMap();
  return impl::mapperElec2Det<ElectronicMapperDummy>(dsElecId2DsDetId);
}

template <>
std::function<std::optional<DsElecId>(DsDetId)>
  createDet2ElecMapper<ElectronicMapperDummy>()
{
  static std::map<uint32_t, uint32_t> dsDetId2dsElecId = impl::inverseMap(buildDsElecId2DsDetIdMap());
  return impl::mapperDet2Elec<ElectronicMapperDummy>(dsDetId2dsElecId);
}

template <>
std::function<std::optional<uint16_t>(FeeLinkId)>
  createFeeLink2SolarMapper<ElectronicMapperDummy>()
{
  static auto f2s = buildFeeLinkId2SolarIdMap();
  return impl::mapperFeeLink2Solar<ElectronicMapperDummy>(f2s);
}

template <>
std::function<std::optional<FeeLinkId>(uint16_t)>
  createSolar2FeeLinkMapper<ElectronicMapperDummy>()
{
  static auto s2f = impl::inverseMap(buildFeeLinkId2SolarIdMap());
  return impl::mapperSolar2FeeLink<ElectronicMapperDummy>(s2f);
}

template <>
std::set<uint16_t> getSolarUIDs<ElectronicMapperDummy>(int deid)
{
  return impl::getSolarUIDs<ElectronicMapperDummy>(deid);
}

template <>
std::set<uint16_t> getSolarUIDs<ElectronicMapperDummy>()
{
  return impl::getSolarUIDs<ElectronicMapperDummy>();
}

template <>
std::vector<std::string> solar2FeeLinkConsistencyCheck<ElectronicMapperDummy>()
{
  return impl::solar2FeeLinkConsistencyCheck<ElectronicMapperDummy>();
}

template <>
std::set<DsElecId> getAllDs<ElectronicMapperDummy>()
{
  return impl::getAllDs<ElectronicMapperDummy>();
}

template <>
std::set<uint16_t> getSolarUIDsPerFeeId<ElectronicMapperDummy>(uint16_t feeid)
{
  return impl::getSolarUIDsPerFeeId<ElectronicMapperDummy>(feeid);
}

template <>
std::set<DsDetId> getDualSampas<ElectronicMapperDummy>(uint16_t solarId)
{
  return impl::getDualSampas<ElectronicMapperDummy>(solarId);
}

template <>
std::set<DsDetId> getDualSampasPerFeeId<ElectronicMapperDummy>(uint16_t feeId)
{
  return impl::getDualSampasPerFeeId<ElectronicMapperDummy>(feeId);
}

} // namespace o2::mch::raw
