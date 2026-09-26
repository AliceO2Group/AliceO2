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

#define BOOST_TEST_MODULE MFT CA Tracking Configuration
#define BOOST_TEST_DYN_LINK
#include "TrackingParameterTestSupport.h"
#include <boost/test/unit_test.hpp>

#include <stdexcept>
#include <string>
#include <TClass.h>
#include <TDataMember.h>

#include "CommonUtils/ConfigurableParam.h"
#include "DetectorsBase/Propagator.h"
#include "ITSMFTTracking/Configuration.h"
#include "ITSMFTTracking/ITSMFTDetectorDefinitions.h"
#include "ITSMFTTracking/TraversalTopology.h"
#include "ITSMFTTracking/IndexTableConfiguration.h"

using namespace o2::itsmft;
using namespace o2::itsmft::tracking;
using o2::conf::ConfigurableParam;
using MFTParam = MFTCATrackerParam;

namespace
{
struct FieldFixture {
  FieldFixture() { o2::base::Propagator::initFieldFromGRP(0.f, 0.f, true, false); }
};
struct RestoreConfiguration {
  ~RestoreConfiguration()
  {
    ConfigurableParam::updateFromString("MFTCATrackerParam.nIterations=-1;MFTCATrackerParam.startLayerMask[0]=0");
  }
};
auto resolve(TrackingMode::Type mode)
{
  return o2::itsmft::tracking::test::referenceTrackingParameters(o2::detectors::DetID::MFT, mode);
}
} // namespace

BOOST_TEST_GLOBAL_FIXTURE(FieldFixture);

BOOST_AUTO_TEST_CASE(DefaultAsyncUsesAllPresetPasses)
{
  BOOST_CHECK_EQUAL(resolve(TrackingMode::Sync).size(), 1);
  BOOST_CHECK_EQUAL(resolve(TrackingMode::Async).size(), 3);
  BOOST_CHECK(resolve(TrackingMode::Off).empty());
  const auto async = resolve(TrackingMode::Async);
  BOOST_CHECK_LT(async[2].TrackletMinPt, async[0].TrackletMinPt);
  BOOST_CHECK_LT(async[2].MinTrackLength, async[0].MinTrackLength);
}

BOOST_FIXTURE_TEST_CASE(ParserPassLimitsAreExplicitAndChecked, RestoreConfiguration)
{
  for (const int count : {1, 2, 3}) {
    ConfigurableParam::updateFromString("MFTCATrackerParam.nIterations=" + std::to_string(count));
    BOOST_CHECK_EQUAL(resolve(TrackingMode::Async).size(), count);
  }
  for (const int count : {0, -2, 4}) {
    ConfigurableParam::updateFromString("MFTCATrackerParam.nIterations=" + std::to_string(count));
    BOOST_CHECK_THROW(resolve(TrackingMode::Async), std::invalid_argument);
  }
  ConfigurableParam::updateFromString("MFTCATrackerParam.nIterations=2");
  BOOST_CHECK_THROW(resolve(TrackingMode::Sync), std::invalid_argument);
  ConfigurableParam::updateFromString("MFTCATrackerParam.nIterations=-1");
  BOOST_CHECK_EQUAL(resolve(TrackingMode::Async).size(), 3);
}

BOOST_FIXTURE_TEST_CASE(ParserOuterLayerMasksReachTheResolvedRoadStarts, RestoreConfiguration)
{
  const DetectorConfiguration layout{kMFTSurfaces};
  for (const auto mode : {TrackingMode::Sync, TrackingMode::Async}) {
    for (const uint32_t mask : {uint32_t{1} << 8, uint32_t{1} << 9, (uint32_t{1} << 8) | (uint32_t{1} << 9)}) {
      ConfigurableParam::updateFromString("MFTCATrackerParam.startLayerMask[0]=" + std::to_string(mask));
      BOOST_CHECK_EQUAL(MFTParam::Instance().startLayerMask[0], mask);
      const auto topology = deriveTraversalTopology(layout, resolve(mode).front());
      BOOST_REQUIRE(topology.ok());
      LayerMask actual;
      for (const auto path : topology.topology->roadStartPaths) {
        const auto edge = topology.topology->paths[path.value()].second;
        actual.set(topology.topology->edges[edge.value()].to.value());
      }
      BOOST_CHECK_EQUAL(actual.value(), mask);
    }
  }
  ConfigurableParam::updateFromString("MFTCATrackerParam.startLayerMask[0]=1024");
  BOOST_CHECK_THROW(resolve(TrackingMode::Async), std::invalid_argument);
  ConfigurableParam::updateFromString("MFTCATrackerParam.startLayerMask[0]=0");
  BOOST_CHECK_EQUAL(resolve(TrackingMode::Sync).front().StartLayerMask.count(), MFTNLayers);
  auto* dictionary = TClass::GetClass(typeid(MFTParam));
  BOOST_REQUIRE(dictionary);
  auto* member = dictionary->GetDataMember("startLayerMask");
  BOOST_REQUIRE(member);
  BOOST_CHECK_EQUAL(member->GetArrayDim(), 1);
  BOOST_CHECK_EQUAL(member->GetMaxIndex(0), MaxIter);
  BOOST_CHECK_EQUAL(member->GetUnitSize(), sizeof(uint32_t));
}

BOOST_AUTO_TEST_CASE(PublicMFTIndexBinsControlRadiusAndPhiLookup)
{
  ConfigurableParam::updateFromString("MFTCATrackerParam.LUTbinsU=32;MFTCATrackerParam.LUTbinsV=24");
  const auto parameters = resolve(TrackingMode::Sync).front();
  std::array<SurfaceChartRange, MFTNLayers> ranges;
  ranges.fill({0.f, 16.f});
  IndexTableUtilsCore index;
  BOOST_REQUIRE(configureIndexTableUtils(index, parameters, MFTNLayers, SurfaceKind::Disk, ranges));
  BOOST_CHECK(index.getCoordType() == IndexTableCoordType::PhiR);
  BOOST_CHECK_EQUAL(index.getRowBinIndex(o2::constants::math::PI), 12);
  BOOST_CHECK_EQUAL(index.getColBinIndex(0, 8.f), 16);
  ConfigurableParam::updateFromString("MFTCATrackerParam.LUTbinsU=64;MFTCATrackerParam.LUTbinsV=128");
}

BOOST_AUTO_TEST_CASE(AsyncOnlyOverridesAreRejectedInOtherActiveModes)
{
  for (const auto* field : {"minTrackLgtIter[0]", "minPtIterLgt[0]"}) {
    const std::string key = std::string{"MFTCATrackerParam."} + field;
    ConfigurableParam::updateFromString(key + "=5");
    BOOST_CHECK_NO_THROW(resolve(TrackingMode::Async));
    for (const auto mode : {TrackingMode::Sync, TrackingMode::Cosmics}) {
      BOOST_CHECK_EXCEPTION(resolve(mode), std::invalid_argument,
                            [&key](const auto& error) { return std::string{error.what()}.find(key.substr(0, key.find('['))) != std::string::npos; });
    }
    ConfigurableParam::updateFromString(key + "=0");
  }
}
