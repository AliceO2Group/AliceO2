// Copyright 2019-2026 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#define BOOST_TEST_MODULE ITSClusterBinRange
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>
#include <array>
#include <cmath>
#include <numbers>
#include <set>
#include <vector>
#include "StrangenessTracking/ITSClusterBinRange.h"
#include "ITStracking/IndexTableUtils.h"

using namespace o2::strangeness_tracking::clustertracking;
namespace
{
struct Params {
  int PhiBins = 128, ZBins = 256;
  std::array<float, 7> LayerZ{17.333f, 17.333f, 17.333f, 43.14f, 43.14f, 74.745f, 74.745f};
};

std::vector<int> phiBins(const BinRange& bins)
{
  std::vector<int> result;
  BOOST_REQUIRE_LE(bins.phiCount(), bins.nPhi);
  for (int i = 0; i < bins.phiCount(); ++i) {
    const int phi = bins.phiBin(i);
    BOOST_REQUIRE_GE(phi, 0);
    BOOST_REQUIRE_LT(phi, bins.nPhi);
    result.push_back(phi);
  }
  BOOST_CHECK_EQUAL(std::set<int>(result.begin(), result.end()).size(), result.size());
  return result;
}
} // namespace

BOOST_AUTO_TEST_CASE(PeriodicLookupTerminatesAndPreservesSupport)
{
  const Params params;
  o2::its::IndexTableUtils<7> utils;
  utils.setTrackingParameters(params);
  // Captured from the nonterminating TF1/PV833/50-MeV lookup: a tiny
  // negative upper angle normalizes to float(2*pi), producing bin 128.
  const auto rect = o2::its::getBinsRect(0, -.0847557857f, 1.3733691f, .511298f, .0847557559f, utils);
  BOOST_REQUIRE_EQUAL(rect.w, params.PhiBins);
  const auto witness = makeBinRange(rect.x, rect.y, rect.z, rect.w, params.ZBins, params.PhiBins, .0847557559f);
  BOOST_CHECK_EQUAL(phiBins(witness).size(), params.PhiBins);

  const auto normal = phiBins(makeBinRange(5, 2, 9, 4, 256, 128, .05f));
  const std::vector<int> expectedNormal{2, 3, 4};
  BOOST_CHECK_EQUAL_COLLECTIONS(normal.begin(), normal.end(), expectedNormal.begin(), expectedNormal.end());
  const auto wrap = phiBins(makeBinRange(5, 124, 9, 2, 256, 128, .15f));
  const std::vector<int> expectedWrap{124, 125, 126, 127, 0, 1, 2};
  BOOST_CHECK_EQUAL_COLLECTIONS(wrap.begin(), wrap.end(), expectedWrap.begin(), expectedWrap.end());
  BOOST_CHECK_EQUAL(phiBins(makeBinRange(5, 2, 9, 4, 256, 128, std::numbers::pi_v<float>)).size(), 128);
  BOOST_CHECK_EQUAL(phiBins(makeBinRange(5, -1, 9, 4, 256, 128, .05f)).size(), 128);
  BOOST_CHECK_EQUAL(phiBins(makeBinRange(5, 128, 9, 4, 256, 128, .05f)).size(), 128);
  const auto upperZ = makeBinRange(256, 2, 255, 4, 256, 128, .05f);
  BOOST_CHECK_EQUAL(upperZ.firstZ, 255);
  BOOST_CHECK_EQUAL(upperZ.lastZ, 255);
  BOOST_CHECK(makeBinRange(-1, -1, -1, -1, 256, 128, .05f).empty());
  BOOST_CHECK(makeBinRange(9, 2, 5, 4, 256, 128, .05f).empty());
  BOOST_CHECK(makeBinRange(0, 0, 1, 1, 0, 128, .05f).empty());
  BOOST_CHECK(makeBinRange(0, 0, 1, 1, 256, 0, .05f).empty());
}

BOOST_AUTO_TEST_CASE(PeriodicLookupDistinguishesNarrowAndNearFullWindows)
{
  const Params params;
  o2::its::IndexTableUtils<7> utils;
  utils.setTrackingParameters(params);

  const auto query = [&](float centre, float halfWidth) {
    const auto rect = o2::its::getBinsRect(0, centre, 0.f, .1f, halfWidth, utils);
    return phiBins(makeBinRange(rect.x, rect.y, rect.z, rect.w, params.ZBins, params.PhiBins, halfWidth));
  };
  constexpr float centre = .7f;
  const auto narrow = query(centre, .001f);
  BOOST_REQUIRE_EQUAL(narrow.size(), 1);
  BOOST_CHECK_EQUAL(narrow.front(), utils.getPhiBinIndex(centre));

  const auto crossesSeam = query(0.f, .01f);
  const std::vector<int> expectedSeam{127, 0};
  BOOST_CHECK_EQUAL_COLLECTIONS(crossesSeam.begin(), crossesSeam.end(), expectedSeam.begin(), expectedSeam.end());

  constexpr float nearFull = 3.13f;
  const auto wrapped = o2::its::getBinsRect(0, centre, 0.f, .1f, nearFull, utils);
  BOOST_REQUIRE_EQUAL(wrapped.y, wrapped.w);
  BOOST_CHECK_NE(wrapped.y, utils.getPhiBinIndex(centre));
  constexpr float pi = std::numbers::pi_v<float>;
  for (float halfWidth : {nearFull, std::nextafter(pi, 0.f), pi, std::nextafter(pi, 4.f)}) {
    BOOST_CHECK_EQUAL(query(centre, halfWidth).size(), params.PhiBins);
  }
}
