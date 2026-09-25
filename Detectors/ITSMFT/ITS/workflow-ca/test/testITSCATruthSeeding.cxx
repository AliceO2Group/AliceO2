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

#define BOOST_TEST_MODULE ITS CA Truth Seeding
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>

#include <array>
#include <vector>

#include "ITSCAWorkflow/TruthSeeding.h"
#include "ITSMFTTracking/ROFLookupTables.h"

using namespace o2::its::ca;
using namespace o2::itsmft::tracking;

BOOST_AUTO_TEST_CASE(ConsecutiveFramesSelectTheirOwnCollisionsAndLookupROFs)
{
  const o2::InteractionRecord firstOrigin{0, 40};
  const std::array<o2::InteractionRecord, 2> collisions{firstOrigin + 50, firstOrigin + 250};
  const std::array<float, 2> z{1.f, 7.f};
  // Two ROFs with nonzero delay and bias, matching the cluster loader.
  const o2::its::LayerTiming timing{.mNROFsTF = 2, .mROFLength = 100, .mROFDelay = 10, .mROFBias = 20};
  for (int frame = 0; frame < 2; ++frame) {
    const auto origin = firstOrigin + 200 * frame;
    const std::array<o2::itsmft::ROFRecord, 2> rofs{{{origin, 0, 0, 0}, {origin + 100, 1, 0, 0}}};
    const auto window = truthSeedingWindow(rofs, origin, timing);
    BOOST_REQUIRE(window);
    BOOST_CHECK_EQUAL(window->lower(), 30u);
    BOOST_CHECK_EQUAL(window->upper(), 230u);
    std::vector<o2::its::Vertex> vertices;
    std::vector<int> eventIds;
    for (int event = 0; event < 2; ++event) {
      if (const auto time = truthSeedingTime(collisions[event], origin, *window, 50)) {
        o2::its::Vertex vertex;
        vertex.setXYZ(0.f, 0.f, z[event]);
        vertex.getTimeStamp() = *time;
        vertices.push_back(vertex);
        eventIds.push_back(event);
      }
    }
    BOOST_REQUIRE_EQUAL(vertices.size(), 1);
    BOOST_CHECK_EQUAL(eventIds.front(), frame);
    BOOST_CHECK_EQUAL(vertices.front().getZ(), z[frame]);
    BOOST_CHECK_EQUAL(vertices.front().getTimeStamp().lower(), 50);
    o2::its::ROFVertexLookupTable<1> lookup;
    lookup.defineLayer(0, 2, 100, 10, 20, 0);
    lookup.init();
    lookup.update(vertices.data(), vertices.size());
    BOOST_CHECK_EQUAL(lookup.getView().getVertices(0, 0).getEntries(), 1);
    BOOST_CHECK_EQUAL(lookup.getView().getVertices(0, 1).getEntries(), 0);
  }
}

BOOST_AUTO_TEST_CASE(TruthTimingPreservesOverlapAndRejectsOutOfFrameEvents)
{
  const o2::InteractionRecord origin{0, 40};
  const o2::its::TimeEstBC window{0, 200};
  const auto overlap = truthSeedingTime(origin - 10, origin, window, 50);
  BOOST_REQUIRE(overlap);
  BOOST_CHECK_EQUAL(overlap->lower(), 0);
  BOOST_CHECK_EQUAL(overlap->upper(), 40);
  BOOST_CHECK(!truthSeedingTime(origin - 50, origin, window, 50));
  BOOST_CHECK(!truthSeedingTime(origin + 200, origin, window, 50));
  BOOST_CHECK(!truthSeedingTime(o2::InteractionRecord{}, origin, window, 50));
  BOOST_CHECK(!truthSeedingTime(origin, origin, window, 0));
  BOOST_CHECK(!truthSeedingTime(origin, origin, {}, 50));
}

BOOST_AUTO_TEST_CASE(TruthWindowUsesActualROFRecordsAndLegacyTimingFields)
{
  const o2::InteractionRecord origin{0, 40};
  const std::array<o2::itsmft::ROFRecord, 2> rofs{{{origin, 0, 0, 0}, {origin + 1000, 1, 0, 0}}};
  o2::its::LayerTiming timing{.mNROFsTF = 2, .mROFLength = 100, .mROFDelay = 10, .mROFBias = 20, .mROFAddTimeErr = 50};
  const auto window = truthSeedingWindow(rofs, origin, timing);
  BOOST_REQUIRE(window);
  BOOST_CHECK_EQUAL(window->lower(), 0u);
  BOOST_CHECK_EQUAL(window->upper(), 1180u);
  BOOST_CHECK(!truthSeedingWindow({}, origin, timing));
  timing.mROFLength = 0;
  BOOST_CHECK(!truthSeedingWindow(rofs, origin, timing));
  timing.mROFLength = std::numeric_limits<uint32_t>::max();
  BOOST_CHECK(!truthSeedingWindow(rofs, origin, timing));
}
