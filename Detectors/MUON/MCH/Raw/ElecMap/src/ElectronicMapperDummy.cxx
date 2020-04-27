// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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

// build the map to go from electronic ds id to detector ds id
std::map<uint32_t, uint32_t> buildDsElecId2DsDetIdMap()
{
  std::map<uint32_t, uint32_t> e2d;

  auto dslist = createDualSampaMapper();

  uint16_t n{0};
  uint16_t solarId{0};
  uint8_t groupId{0};
  uint8_t index{0};

  for (auto deId : o2::mch::raw::deIdsForAllMCH) {
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
  uint16_t solarId{0};

  auto dslist = createDualSampaMapper();

  for (auto deId : o2::mch::raw::deIdsForAllMCH) {
    // assign a tuple (solarId,groupId,index) to the pair (deId,dsId)
    for (auto dsId : dslist(deId)) {
      if (n % 40 == 0) {
        solarId++;
        auto cruId = solarId / 24;
        auto linkId = solarId - cruId * 24;
        c2s[encode(o2::mch::raw::FeeLinkId(cruId, linkId))] = solarId;
      }
      n++;
    };
  };
  auto cruId = solarId / 24;
  auto linkId = solarId - cruId * 24;
  c2s[encode(o2::mch::raw::FeeLinkId(cruId, linkId))] = solarId;
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
} // namespace o2::mch::raw
