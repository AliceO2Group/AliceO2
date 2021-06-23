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
#include "ElectronicMapperImplHelper.h"
#include <map>
#include "MCHRawElecMap/DsElecId.h"
#include "MCHRawElecMap/DsDetId.h"
#include <iostream>
#include <fmt/format.h>

extern void fillElec2DetCH5R(std::map<uint32_t, uint32_t>& e2d);
extern void fillElec2DetCH5L(std::map<uint32_t, uint32_t>& e2d);
extern void fillElec2DetCH6R(std::map<uint32_t, uint32_t>& e2d);
extern void fillElec2DetCH6L(std::map<uint32_t, uint32_t>& e2d);
extern void fillElec2DetCH7R(std::map<uint32_t, uint32_t>& e2d);
extern void fillElec2DetCH7L(std::map<uint32_t, uint32_t>& e2d);
extern void fillElec2DetCH8L(std::map<uint32_t, uint32_t>& e2d);
extern void fillElec2DetCH8R(std::map<uint32_t, uint32_t>& e2d);
extern void fillElec2DetCH9L(std::map<uint32_t, uint32_t>& e2d);
extern void fillElec2DetCH9R(std::map<uint32_t, uint32_t>& e2d);
extern void fillElec2DetCH10L(std::map<uint32_t, uint32_t>& e2d);
extern void fillElec2DetCH10R(std::map<uint32_t, uint32_t>& e2d);

extern void fillSolar2FeeLinkCH5R(std::map<uint16_t, uint32_t>& s2c);
extern void fillSolar2FeeLinkCH5L(std::map<uint16_t, uint32_t>& s2c);
extern void fillSolar2FeeLinkCH6R(std::map<uint16_t, uint32_t>& s2c);
extern void fillSolar2FeeLinkCH6L(std::map<uint16_t, uint32_t>& s2c);
extern void fillSolar2FeeLinkCH7R(std::map<uint16_t, uint32_t>& s2c);
extern void fillSolar2FeeLinkCH7L(std::map<uint16_t, uint32_t>& s2c);
extern void fillSolar2FeeLinkCH8L(std::map<uint16_t, uint32_t>& s2c);
extern void fillSolar2FeeLinkCH8R(std::map<uint16_t, uint32_t>& s2c);
extern void fillSolar2FeeLinkCH9L(std::map<uint16_t, uint32_t>& s2c);
extern void fillSolar2FeeLinkCH9R(std::map<uint16_t, uint32_t>& s2c);
extern void fillSolar2FeeLinkCH10L(std::map<uint16_t, uint32_t>& s2c);
extern void fillSolar2FeeLinkCH10R(std::map<uint16_t, uint32_t>& s2c);

namespace
{

void dump(const std::map<uint32_t, uint32_t>& e2d)
{
  for (auto p : e2d) {
    std::cout << o2::mch::raw::decodeDsElecId(p.first).value() << " -> " << o2::mch::raw::decodeDsDetId(p.second) << "\n";
  }
}

std::map<uint32_t, uint32_t> buildDsElecId2DsDetIdMap()
{
  std::map<uint32_t, uint32_t> e2d;
  fillElec2DetCH5R(e2d);
  fillElec2DetCH5L(e2d);
  fillElec2DetCH6R(e2d);
  fillElec2DetCH6L(e2d);
  fillElec2DetCH7R(e2d);
  fillElec2DetCH7L(e2d);
  fillElec2DetCH8R(e2d);
  fillElec2DetCH8L(e2d);
  fillElec2DetCH9R(e2d);
  fillElec2DetCH9L(e2d);
  fillElec2DetCH10R(e2d);
  fillElec2DetCH10L(e2d);
  return e2d;
}

std::map<uint16_t, uint32_t> buildSolarId2FeeLinkIdMap()
{
  std::map<uint16_t, uint32_t> s2f;
  fillSolar2FeeLinkCH5R(s2f);
  fillSolar2FeeLinkCH5L(s2f);
  fillSolar2FeeLinkCH6R(s2f);
  fillSolar2FeeLinkCH6L(s2f);
  fillSolar2FeeLinkCH7R(s2f);
  fillSolar2FeeLinkCH7L(s2f);
  fillSolar2FeeLinkCH8R(s2f);
  fillSolar2FeeLinkCH8L(s2f);
  fillSolar2FeeLinkCH9R(s2f);
  fillSolar2FeeLinkCH9L(s2f);
  fillSolar2FeeLinkCH10R(s2f);
  fillSolar2FeeLinkCH10L(s2f);
  return s2f;
}

} // namespace

namespace o2::mch::raw
{

template <>
std::function<std::optional<DsDetId>(DsElecId)>
  createElec2DetMapper<ElectronicMapperGenerated>(uint64_t /*timestamp*/)
{
  static std::map<uint32_t, uint32_t> dsElecId2DsDetId = buildDsElecId2DsDetIdMap();
  return impl::mapperElec2Det<ElectronicMapperGenerated>(dsElecId2DsDetId);
}

template <>
std::function<std::optional<DsElecId>(DsDetId)>
  createDet2ElecMapper<ElectronicMapperGenerated>()
{
  static std::map<uint32_t, uint32_t> dsDetId2dsElecId = impl::inverseMap(buildDsElecId2DsDetIdMap());
  return impl::mapperDet2Elec<ElectronicMapperGenerated>(dsDetId2dsElecId);
}

template <>
std::function<std::optional<FeeLinkId>(uint16_t)>
  createSolar2FeeLinkMapper<ElectronicMapperGenerated>()
{
  static std::map<uint16_t, uint32_t> solarId2FeeLinkId = buildSolarId2FeeLinkIdMap();
  return impl::mapperSolar2FeeLink<ElectronicMapperGenerated>(solarId2FeeLinkId);
}

template <>
std::function<std::optional<uint16_t>(FeeLinkId)>
  createFeeLink2SolarMapper<ElectronicMapperGenerated>()
{
  static std::map<uint32_t, uint16_t> feeLinkId2SolarId = impl::inverseMap(buildSolarId2FeeLinkIdMap());
  return impl::mapperFeeLink2Solar<ElectronicMapperGenerated>(feeLinkId2SolarId);
}

template <>
std::set<uint16_t> getSolarUIDs<ElectronicMapperGenerated>(int deid)
{
  return impl::getSolarUIDs<ElectronicMapperGenerated>(deid);
}

template <>
std::set<uint16_t> getSolarUIDs<ElectronicMapperGenerated>()
{
  return impl::getSolarUIDs<ElectronicMapperGenerated>();
}

template <>
std::vector<std::string> solar2FeeLinkConsistencyCheck<ElectronicMapperGenerated>()
{
  return impl::solar2FeeLinkConsistencyCheck<ElectronicMapperGenerated>();
}

template <>
std::set<DsElecId> getAllDs<ElectronicMapperGenerated>()
{
  return impl::getAllDs<ElectronicMapperGenerated>();
}

} // namespace o2::mch::raw
