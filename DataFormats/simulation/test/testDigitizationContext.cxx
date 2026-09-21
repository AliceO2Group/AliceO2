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

#define BOOST_TEST_MODULE Test DigitizationContext class
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>
#include "SimulationDataFormat/DigitizationContext.h"
#include <vector>

namespace o2
{

// build a context whose collisions sit at the given orbits (one collision each, source 0)
steer::DigitizationContext makeContext(std::vector<long> const& orbits)
{
  steer::DigitizationContext ctx;
  auto& records = ctx.getEventRecords();
  auto& parts = ctx.getEventParts();
  int entry = 0;
  for (auto o : orbits) {
    records.emplace_back(o2::InteractionTimeRecord(o2::InteractionRecord(0, o), 0.));
    parts.push_back({steer::EventPart(0, entry++)});
  }
  ctx.setNCollisions(records.size());
  ctx.setMaxNumberParts(1);
  return ctx;
}

// The timeframe index structure must have one entry per timeframe asked for, and entry i must
// describe exactly the collisions falling into orbits [start + i*orbitsPerTF, start + (i+1)*orbitsPerTF).
BOOST_AUTO_TEST_CASE(TimeframeIndicesAreSlotAligned)
{
  long const orbitsPerTF = 6;
  long const start = 0;
  long const nTF = 5; // orbits 0..29

  // timeframe 1 (orbits 6..11) and timeframe 4 (orbits 24..29) hold no collision
  std::vector<long> orbits{0, 3, 5, 12, 14, 17, 18, 21};
  auto ctx = makeContext(orbits);

  auto indices = ctx.calcTimeframeIndices(start, orbitsPerTF, 0., nTF);
  BOOST_CHECK_EQUAL(indices.size(), (size_t)nTF);

  for (int tf = 0; tf < nTF; ++tf) {
    auto first = std::get<0>(indices[tf]);
    auto last = std::get<1>(indices[tf]);
    long const lo = start + tf * orbitsPerTF;
    long const hi = lo + orbitsPerTF;
    // count what should be in this timeframe
    int expected = 0;
    for (auto o : orbits) {
      if (o >= lo && o < hi) {
        expected++;
      }
    }
    BOOST_CHECK_EQUAL(last - first + 1, expected);
    for (int i = first; i <= last; ++i) {
      BOOST_CHECK(orbits[i] >= lo);
      BOOST_CHECK(orbits[i] < hi);
    }
  }
}

// A timeframe without collisions must survive extraction as a valid, empty context
BOOST_AUTO_TEST_CASE(EmptyTimeframeExtracts)
{
  long const orbitsPerTF = 6;
  long const nTF = 3;
  auto ctx = makeContext({0, 2, 13}); // timeframe 1 (orbits 6..11) is empty
  auto indices = ctx.calcTimeframeIndices(0, orbitsPerTF, 0., nTF);
  BOOST_CHECK_EQUAL(indices.size(), (size_t)nTF);

  auto tf0 = ctx.extractSingleTimeframe(0, indices, {});
  auto tf1 = ctx.extractSingleTimeframe(1, indices, {});
  auto tf2 = ctx.extractSingleTimeframe(2, indices, {});
  BOOST_CHECK_EQUAL(tf0.getEventRecords().size(), (size_t)2);
  BOOST_CHECK_EQUAL(tf1.getEventRecords().size(), (size_t)0);
  BOOST_CHECK_EQUAL(tf2.getEventRecords().size(), (size_t)1);
  BOOST_CHECK_EQUAL(tf2.getEventRecords()[0].orbit, 13);
}

// The trailing timeframes of the requested range must be present even when the last collision
// falls well before the end of the range
BOOST_AUTO_TEST_CASE(TrailingTimeframesArePresent)
{
  long const orbitsPerTF = 6;
  long const nTF = 9; // this is what an 8-timeframe anchored MC job with orbitsEarly asks for
  auto ctx = makeContext({1, 2, 7});
  auto indices = ctx.calcTimeframeIndices(0, orbitsPerTF, 0., nTF);
  BOOST_CHECK_EQUAL(indices.size(), (size_t)nTF);
  for (int tf = 2; tf < nTF; ++tf) {
    BOOST_CHECK(std::get<0>(indices[tf]) > std::get<1>(indices[tf])); // empty, but present
  }
}

// applyMaxCollisionFilter must not shift timeframes when one of them is empty
BOOST_AUTO_TEST_CASE(MaxCollisionFilterKeepsSlots)
{
  long const orbitsPerTF = 6;
  long const nTF = 4;
  //  tf0: orbits 0,1,2   tf1: empty   tf2: orbits 12,13   tf3: orbit 19
  auto ctx = makeContext({0, 1, 2, 12, 13, 19});
  auto indices = ctx.calcTimeframeIndices(0, orbitsPerTF, 0., nTF);
  ctx.applyMaxCollisionFilter(indices, 0, orbitsPerTF, 2, 0.); // keep at most 2 per timeframe

  BOOST_CHECK_EQUAL(indices.size(), (size_t)nTF);
  BOOST_CHECK_EQUAL(std::get<1>(indices[0]) - std::get<0>(indices[0]) + 1, 2); // capped
  BOOST_CHECK(std::get<0>(indices[1]) > std::get<1>(indices[1]));              // still empty
  BOOST_CHECK_EQUAL(std::get<1>(indices[2]) - std::get<0>(indices[2]) + 1, 2);
  BOOST_CHECK_EQUAL(std::get<1>(indices[3]) - std::get<0>(indices[3]) + 1, 1);

  auto tf2 = ctx.extractSingleTimeframe(2, indices, {});
  BOOST_CHECK_EQUAL(tf2.getEventRecords().size(), (size_t)2);
  BOOST_CHECK_EQUAL(tf2.getEventRecords()[0].orbit, 12);
}

} // namespace o2
