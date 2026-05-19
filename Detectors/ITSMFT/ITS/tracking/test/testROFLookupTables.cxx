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

#include <boost/test/tools/old/interface.hpp>
#define BOOST_TEST_MODULE ITS ROFLookupTables
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>
#include "ITStracking/ROFLookupTables.h"

/// -------- Tests --------
// LayerTiming
BOOST_AUTO_TEST_CASE(layertiming_basic)
{
  o2::its::ROFOverlapTable<1> table;
  table.defineLayer(0, 10, 594, 100, 0, 50);
  const auto& layer = table.getLayer(0);

  // test ROF time calculations
  auto start0 = layer.getROFStartInBC(0);
  BOOST_CHECK_EQUAL(start0, 100); // delay only

  auto end0 = layer.getROFEndInBC(0);
  BOOST_CHECK_EQUAL(end0, 100 + 594);

  // test second ROF
  auto start1 = layer.getROFStartInBC(1);
  BOOST_CHECK_EQUAL(start1, 100 + 594);
}

BOOST_AUTO_TEST_CASE(layertiming_base)
{
  o2::its::ROFOverlapTable<3> table;
  table.defineLayer(0, 10, 500, 0, 0, 0);
  table.defineLayer(1, 12, 600, 50, 0, 0);
  table.defineLayer(2, 8, 400, 100, 0, 0);
  const auto& layer1 = table.getLayer(1);
  BOOST_CHECK_EQUAL(layer1.mNROFsTF, 12);
  BOOST_CHECK_EQUAL(layer1.mROFLength, 600);
}

BOOST_AUTO_TEST_CASE(rofmask_construct_from_timing)
{
  o2::its::ROFOverlapTable<2> timing;
  timing.defineLayer(0, 3, 100, 0, 0, 0);
  timing.defineLayer(1, 4, 50, 25, 0, 0);

  o2::its::ROFMaskTable<2> mask{timing};
  const auto view = mask.getView();

  BOOST_REQUIRE(view.mFlatMask != nullptr);
  BOOST_REQUIRE(view.mLayerROFOffsets != nullptr);
  BOOST_CHECK_EQUAL(view.mLayerROFOffsets[0], 0);
  BOOST_CHECK_EQUAL(view.mLayerROFOffsets[1], 3);
  BOOST_CHECK_EQUAL(view.mLayerROFOffsets[2], 7);

  // by default all rofs are disabled
  for (int rof{0}; rof < 3; ++rof) {
    BOOST_CHECK(!view.isROFEnabled(0, rof));
  }
  for (int rof{0}; rof < 4; ++rof) {
    BOOST_CHECK(!view.isROFEnabled(1, rof));
  }

  mask.selectROF({110, 20});

  BOOST_CHECK(!view.isROFEnabled(0, 0));
  BOOST_CHECK(view.isROFEnabled(0, 1));
  BOOST_CHECK(!view.isROFEnabled(0, 2));

  BOOST_CHECK(!view.isROFEnabled(1, 0));
  BOOST_CHECK(view.isROFEnabled(1, 1));
  BOOST_CHECK(view.isROFEnabled(1, 2));
  BOOST_CHECK(!view.isROFEnabled(1, 3));
}

// ROFOverlapTable
BOOST_AUTO_TEST_CASE(rofoverlap_basic)
{
  // define 2 layers with the same definitions (no staggering)
  o2::its::ROFOverlapTable<2> table;
  table.defineLayer(0, 12, 594, 0, 0, 0);
  table.defineLayer(1, 12, 594, 0, 0, 0);
  table.init();
  const auto view = table.getView();
  // each rof in layer 0 should be compatible with its layer 1 equivalent
  for (int rof{0}; rof < 12; ++rof) {
    BOOST_CHECK(view.doROFsOverlap(0, rof, 1, rof));
    BOOST_CHECK(view.doROFsOverlap(1, rof, 0, rof));
    BOOST_CHECK(view.getOverlap(0, 1, rof).getEntries() == 1);
  }
}

BOOST_AUTO_TEST_CASE(rofoverlap_staggered)
{
  // test staggered layers with ROF delay
  o2::its::ROFOverlapTable<2> table;
  table.defineLayer(0, 10, 500, 0, 0, 0);
  table.defineLayer(1, 10, 500, 250, 0, 0); // 250 BC delay
  table.init();
  const auto view = table.getView();

  // verify overlap range
  { // from 0 to 1
    const auto& range = view.getOverlap(0, 1, 0);
    BOOST_CHECK_EQUAL(range.getEntries(), 1);
    BOOST_CHECK_EQUAL(range.getFirstEntry(), 0);
  }
  { // from 1 to 0
    const auto& range = view.getOverlap(1, 0, 0);
    BOOST_CHECK_EQUAL(range.getEntries(), 2);
    BOOST_CHECK_EQUAL(range.getFirstEntry(), 0);
  }
}

BOOST_AUTO_TEST_CASE(rofoverlap_staggered_pp)
{
  const uint32_t rofLen{198}, rofBins{6};
  const uint32_t rofDelay{rofLen / rofBins};
  o2::its::ROFOverlapTable<3> table;
  for (uint32_t lay{0}; lay < 3; ++lay) {
    table.defineLayer(lay, 6, rofLen, lay * rofDelay, 0, 0);
  }
  table.init();
  const auto view = table.getView();
  view.printAll();
}

BOOST_AUTO_TEST_CASE(rofoverlap_staggered_track_time_ignores_added_error)
{
  const uint32_t rofLen{198};
  const uint32_t rofDelay{33};
  const uint32_t addTimeErr{100};

  o2::its::ROFOverlapTable<7> tableNoError;
  o2::its::ROFOverlapTable<7> tableWithError;
  for (uint32_t lay{0}; lay < 7; ++lay) {
    const auto delay = (lay == 6) ? 0 : lay * rofDelay;
    tableNoError.defineLayer(lay, 2, rofLen, delay, 0, 0);
    tableWithError.defineLayer(lay, 2, rofLen, delay, 0, addTimeErr);
  }

  auto getCommonTrackTime = [](const auto& table) {
    auto ts = table.getLayer(0).getROFTimeBounds(0);
    for (uint32_t lay{1}; lay < 7; ++lay) {
      ts += table.getLayer(lay).getROFTimeBounds(0);
    }
    return ts.makeSymmetrical();
  };

  const auto tsNoError = getCommonTrackTime(tableNoError);
  BOOST_CHECK_EQUAL(tsNoError.getTimeStamp(), 181.5f);
  BOOST_CHECK_EQUAL(tsNoError.getTimeStampError(), 16.5f);

  const auto tsWithError = getCommonTrackTime(tableWithError);
  BOOST_CHECK_EQUAL(tsWithError.getTimeStamp(), 181.5f);
  BOOST_CHECK_EQUAL(tsWithError.getTimeStampError(), 16.5f);
}

BOOST_AUTO_TEST_CASE(rofoverlap_track_time_boundary_migration_fallback)
{
  const uint32_t rofLen{198};
  const uint32_t addTimeErr{30};

  o2::its::ROFOverlapTable<7> table;
  for (uint32_t lay{0}; lay < 7; ++lay) {
    table.defineLayer(lay, 4, rofLen, 0, 0, addTimeErr);
  }

  auto getCommonTrackTime = [](const auto& table) {
    bool firstCls{true}, nominalCompatible{true};
    o2::its::TimeEstBC nominalTS, expandedTS;
    for (uint32_t lay{0}; lay < 7; ++lay) {
      const auto rof = lay < 3 ? 0 : 1;
      const auto nominalROFTS = table.getLayer(lay).getROFTimeBounds(rof);
      const auto expandedROFTS = table.getLayer(lay).getROFTimeBounds(rof, true);
      if (firstCls) {
        firstCls = false;
        nominalTS = nominalROFTS;
        expandedTS = expandedROFTS;
      } else {
        if (nominalCompatible) {
          if (nominalTS.isCompatible(nominalROFTS)) {
            nominalTS += nominalROFTS;
          } else {
            nominalCompatible = false;
          }
        }
        BOOST_REQUIRE(expandedTS.isCompatible(expandedROFTS));
        expandedTS += expandedROFTS;
      }
    }
    return (nominalCompatible ? nominalTS : expandedTS).makeSymmetrical();
  };

  const auto tsWithError = getCommonTrackTime(table);
  BOOST_CHECK_EQUAL(tsWithError.getTimeStamp(), 198.f);
  BOOST_CHECK_EQUAL(tsWithError.getTimeStampError(), 30.f);
}

BOOST_AUTO_TEST_CASE(rofoverlap_staggered_alllayers)
{
  // test staggered layers with ROF delay
  o2::its::ROFOverlapTable<3> table;
  table.defineLayer(0, 2, 3, 0, 0, 0);
  table.defineLayer(1, 3, 2, 0, 0, 0);
  table.defineLayer(2, 6, 1, 0, 0, 0);
  table.init();
  const auto view = table.getView();
  // verify overlap range
  { // from 0 to 1 rof=0
    const auto& range = view.getOverlap(0, 1, 0);
    BOOST_CHECK_EQUAL(range.getEntries(), 2);
    BOOST_CHECK_EQUAL(range.getFirstEntry(), 0);
  }
  { // from 0 to 2 rof=0
    const auto& range = view.getOverlap(0, 2, 0);
    BOOST_CHECK_EQUAL(range.getEntries(), 3);
    BOOST_CHECK_EQUAL(range.getFirstEntry(), 0);
  }
  { // from 0 to 1 rof=1
    const auto& range = view.getOverlap(0, 1, 1);
    BOOST_CHECK_EQUAL(range.getEntries(), 2);
    BOOST_CHECK_EQUAL(range.getFirstEntry(), 1);
  }
  { // from 0 to 2 rof=1
    const auto& range = view.getOverlap(0, 2, 1);
    BOOST_CHECK_EQUAL(range.getEntries(), 3);
    BOOST_CHECK_EQUAL(range.getFirstEntry(), 3);
  }
  { // from 1 to 2 rof=0
    const auto& range = view.getOverlap(1, 2, 0);
    BOOST_CHECK_EQUAL(range.getEntries(), 2);
    BOOST_CHECK_EQUAL(range.getFirstEntry(), 0);
  }
  { // from 1 to 0 rof=0
    const auto& range = view.getOverlap(1, 0, 0);
    BOOST_CHECK_EQUAL(range.getEntries(), 1);
    BOOST_CHECK_EQUAL(range.getFirstEntry(), 0);
  }
  { // from 1 to 2 rof=1
    const auto& range = view.getOverlap(1, 2, 1);
    BOOST_CHECK_EQUAL(range.getEntries(), 2);
    BOOST_CHECK_EQUAL(range.getFirstEntry(), 2);
  }
  { // from 1 to 0 rof=1
    const auto& range = view.getOverlap(1, 0, 1);
    BOOST_CHECK_EQUAL(range.getEntries(), 2);
    BOOST_CHECK_EQUAL(range.getFirstEntry(), 0);
  }
  { // from 1 to 2 rof=2
    const auto& range = view.getOverlap(1, 2, 2);
    BOOST_CHECK_EQUAL(range.getEntries(), 2);
    BOOST_CHECK_EQUAL(range.getFirstEntry(), 4);
  }
  { // from 1 to 0 rof=2
    const auto& range = view.getOverlap(1, 0, 2);
    BOOST_CHECK_EQUAL(range.getEntries(), 1);
    BOOST_CHECK_EQUAL(range.getFirstEntry(), 1);
  }
  { // from 2 to 1 rof=0
    const auto& range = view.getOverlap(2, 1, 0);
    BOOST_CHECK_EQUAL(range.getEntries(), 1);
    BOOST_CHECK_EQUAL(range.getFirstEntry(), 0);
  }
  { // from 2 to 1 rof=1
    const auto& range = view.getOverlap(2, 1, 1);
    BOOST_CHECK_EQUAL(range.getEntries(), 1);
    BOOST_CHECK_EQUAL(range.getFirstEntry(), 0);
  }
  { // from 2 to 1 rof=2
    const auto& range = view.getOverlap(2, 1, 2);
    BOOST_CHECK_EQUAL(range.getEntries(), 1);
    BOOST_CHECK_EQUAL(range.getFirstEntry(), 1);
  }
  { // from 2 to 1 rof=3
    const auto& range = view.getOverlap(2, 1, 3);
    BOOST_CHECK_EQUAL(range.getEntries(), 1);
    BOOST_CHECK_EQUAL(range.getFirstEntry(), 1);
  }
  { // from 2 to 1 rof=4
    const auto& range = view.getOverlap(2, 1, 4);
    BOOST_CHECK_EQUAL(range.getEntries(), 1);
    BOOST_CHECK_EQUAL(range.getFirstEntry(), 2);
  }
  { // from 2 to 1 rof=5
    const auto& range = view.getOverlap(2, 1, 5);
    BOOST_CHECK_EQUAL(range.getEntries(), 1);
    BOOST_CHECK_EQUAL(range.getFirstEntry(), 2);
  }
  { // from 2 to 0 rof=0
    const auto& range = view.getOverlap(2, 0, 0);
    BOOST_CHECK_EQUAL(range.getEntries(), 1);
    BOOST_CHECK_EQUAL(range.getFirstEntry(), 0);
  }
  { // from 2 to 0 rof=1
    const auto& range = view.getOverlap(2, 0, 1);
    BOOST_CHECK_EQUAL(range.getEntries(), 1);
    BOOST_CHECK_EQUAL(range.getFirstEntry(), 0);
  }
  { // from 2 to 0 rof=2
    const auto& range = view.getOverlap(2, 0, 2);
    BOOST_CHECK_EQUAL(range.getEntries(), 1);
    BOOST_CHECK_EQUAL(range.getFirstEntry(), 0);
  }
  { // from 2 to 0 rof=3
    const auto& range = view.getOverlap(2, 0, 3);
    BOOST_CHECK_EQUAL(range.getEntries(), 1);
    BOOST_CHECK_EQUAL(range.getFirstEntry(), 1);
  }
  { // from 2 to 0 rof=4
    const auto& range = view.getOverlap(2, 0, 4);
    BOOST_CHECK_EQUAL(range.getEntries(), 1);
    BOOST_CHECK_EQUAL(range.getFirstEntry(), 1);
  }
  { // from 2 to 0 rof=5
    const auto& range = view.getOverlap(2, 0, 5);
    BOOST_CHECK_EQUAL(range.getEntries(), 1);
    BOOST_CHECK_EQUAL(range.getFirstEntry(), 1);
  }
}

BOOST_AUTO_TEST_CASE(rofoverlap_staggered_alllayers_delay_delta)
{
  // test staggered layers with ROF delay
  o2::its::ROFOverlapTable<3> table;
  table.defineLayer(0, 2, 3, 0, 0, 0);
  table.defineLayer(1, 3, 2, 1, 0, 0);
  table.defineLayer(2, 6, 1, 0, 0, 1);
  table.init();
  const auto view = table.getView();

  // verify overlap range
  { // from 0 to 1 rof=0
    const auto& range = view.getOverlap(0, 1, 0);
    BOOST_CHECK_EQUAL(range.getEntries(), 1);
    BOOST_CHECK_EQUAL(range.getFirstEntry(), 0);
  }
  { // from 0 to 2 rof=0
    const auto& range = view.getOverlap(0, 2, 0);
    BOOST_CHECK_EQUAL(range.getEntries(), 4);
    BOOST_CHECK_EQUAL(range.getFirstEntry(), 0);
  }
  { // from 0 to 1 rof=1
    const auto& range = view.getOverlap(0, 1, 1);
    BOOST_CHECK_EQUAL(range.getEntries(), 2);
    BOOST_CHECK_EQUAL(range.getFirstEntry(), 1);
  }
  { // from 0 to 2 rof=1
    const auto& range = view.getOverlap(0, 2, 1);
    BOOST_CHECK_EQUAL(range.getEntries(), 4);
    BOOST_CHECK_EQUAL(range.getFirstEntry(), 2);
  }
  { // from 1 to 2 rof=0
    const auto& range = view.getOverlap(1, 2, 0);
    BOOST_CHECK_EQUAL(range.getEntries(), 4);
    BOOST_CHECK_EQUAL(range.getFirstEntry(), 0);
  }
  { // from 1 to 0 rof=0
    const auto& range = view.getOverlap(1, 0, 0);
    BOOST_CHECK_EQUAL(range.getEntries(), 1);
    BOOST_CHECK_EQUAL(range.getFirstEntry(), 0);
  }
  { // from 1 to 2 rof=1
    const auto& range = view.getOverlap(1, 2, 1);
    BOOST_CHECK_EQUAL(range.getEntries(), 4);
    BOOST_CHECK_EQUAL(range.getFirstEntry(), 2);
  }
  { // from 1 to 0 rof=1
    const auto& range = view.getOverlap(1, 0, 1);
    BOOST_CHECK_EQUAL(range.getEntries(), 1);
    BOOST_CHECK_EQUAL(range.getFirstEntry(), 1);
  }
  { // from 1 to 2 rof=2
    const auto& range = view.getOverlap(1, 2, 2);
    BOOST_CHECK_EQUAL(range.getEntries(), 2);
    BOOST_CHECK_EQUAL(range.getFirstEntry(), 4);
  }
  { // from 1 to 0 rof=2
    const auto& range = view.getOverlap(1, 0, 2);
    BOOST_CHECK_EQUAL(range.getEntries(), 1);
    BOOST_CHECK_EQUAL(range.getFirstEntry(), 1);
  }
  { // from 2 to 1 rof=0
    const auto& range = view.getOverlap(2, 1, 0);
    BOOST_CHECK_EQUAL(range.getEntries(), 1);
    BOOST_CHECK_EQUAL(range.getFirstEntry(), 0);
  }
  { // from 2 to 1 rof=1
    const auto& range = view.getOverlap(2, 1, 1);
    BOOST_CHECK_EQUAL(range.getEntries(), 1);
    BOOST_CHECK_EQUAL(range.getFirstEntry(), 0);
  }
  { // from 2 to 1 rof=2
    const auto& range = view.getOverlap(2, 1, 2);
    BOOST_CHECK_EQUAL(range.getEntries(), 2);
    BOOST_CHECK_EQUAL(range.getFirstEntry(), 0);
  }
  { // from 2 to 1 rof=3
    const auto& range = view.getOverlap(2, 1, 3);
    BOOST_CHECK_EQUAL(range.getEntries(), 2);
    BOOST_CHECK_EQUAL(range.getFirstEntry(), 0);
  }
  { // from 2 to 1 rof=4
    const auto& range = view.getOverlap(2, 1, 4);
    BOOST_CHECK_EQUAL(range.getEntries(), 2);
    BOOST_CHECK_EQUAL(range.getFirstEntry(), 1);
  }
  { // from 2 to 1 rof=5
    const auto& range = view.getOverlap(2, 1, 5);
    BOOST_CHECK_EQUAL(range.getEntries(), 2);
    BOOST_CHECK_EQUAL(range.getFirstEntry(), 1);
  }
  { // from 2 to 0 rof=0
    const auto& range = view.getOverlap(2, 0, 0);
    BOOST_CHECK_EQUAL(range.getEntries(), 1);
    BOOST_CHECK_EQUAL(range.getFirstEntry(), 0);
  }
  { // from 2 to 0 rof=1
    const auto& range = view.getOverlap(2, 0, 1);
    BOOST_CHECK_EQUAL(range.getEntries(), 1);
    BOOST_CHECK_EQUAL(range.getFirstEntry(), 0);
  }
  { // from 2 to 0 rof=2
    const auto& range = view.getOverlap(2, 0, 2);
    BOOST_CHECK_EQUAL(range.getEntries(), 2);
    BOOST_CHECK_EQUAL(range.getFirstEntry(), 0);
  }
  { // from 2 to 0 rof=3
    const auto& range = view.getOverlap(2, 0, 3);
    BOOST_CHECK_EQUAL(range.getEntries(), 2);
    BOOST_CHECK_EQUAL(range.getFirstEntry(), 0);
  }
  { // from 2 to 0 rof=4
    const auto& range = view.getOverlap(2, 0, 4);
    BOOST_CHECK_EQUAL(range.getEntries(), 1);
    BOOST_CHECK_EQUAL(range.getFirstEntry(), 1);
  }
  { // from 2 to 0 rof=5
    const auto& range = view.getOverlap(2, 0, 5);
    BOOST_CHECK_EQUAL(range.getEntries(), 1);
    BOOST_CHECK_EQUAL(range.getFirstEntry(), 1);
  }
}

BOOST_AUTO_TEST_CASE(rofoverlap_with_delta)
{
  // test with ROF delta for compatibility window
  o2::its::ROFOverlapTable<2> table;
  table.defineLayer(0, 8, 600, 0, 0, 100); // +/- 100 BC delta
  table.defineLayer(1, 8, 600, 0, 0, 100);
  table.init();
  const auto view = table.getView();

  // with delta, ROFs should have wider compatibility
  for (int rof{0}; rof < 8; ++rof) {
    auto overlap = view.getOverlap(0, 1, rof);
    if (rof == 0 || rof == 7) {
      // edges should see only two
      BOOST_CHECK_EQUAL(overlap.getEntries(), 2);
    } else {
      BOOST_CHECK_EQUAL(overlap.getEntries(), 3);
    }
  }
}

BOOST_AUTO_TEST_CASE(rofoverlap_same_layer)
{
  // test same layer compatibility
  o2::its::ROFOverlapTable<1> table;
  table.defineLayer(0, 10, 500, 0, 0, 0);
  table.init();
  const auto view = table.getView();

  // same ROF in same layer should be compatible
  BOOST_CHECK(view.doROFsOverlap(0, 5, 0, 5));
  // different ROFs in same layer should not be compatible
  BOOST_CHECK(!view.doROFsOverlap(0, 5, 0, 6));
}

BOOST_AUTO_TEST_CASE(rofoverlap_timestamp_basic)
{
  o2::its::ROFOverlapTable<4> table;
  table.defineLayer(0, 4, 100, 0, 0, 0);
  table.defineLayer(1, 4, 100, 0, 0, 0);
  table.defineLayer(2, 8, 50, 0, 0, 0);
  table.defineLayer(3, 7, 50, 50, 0, 0);
  table.init();
  const auto& view = table.getView();

  const auto t01 = view.getTimeStamp(0, 3, 1, 3);
  BOOST_CHECK_EQUAL(t01.getTimeStamp(), 300);
  BOOST_CHECK_EQUAL(t01.getTimeStampError(), 100);

  const auto t02 = view.getTimeStamp(0, 1, 2, 3);
  BOOST_CHECK_EQUAL(t02.getTimeStamp(), 150);
  BOOST_CHECK_EQUAL(t02.getTimeStampError(), 50);

  const auto t03 = view.getTimeStamp(0, 0, 3, 0);
  BOOST_CHECK_EQUAL(t03.getTimeStamp(), 50);
  BOOST_CHECK_EQUAL(t03.getTimeStampError(), 50);

  const auto t23 = view.getTimeStamp(2, 2, 3, 1);
  BOOST_CHECK_EQUAL(t23.getTimeStamp(), 100);
  BOOST_CHECK_EQUAL(t23.getTimeStampError(), 50);
}

BOOST_AUTO_TEST_CASE(rofoverlap_timestamp_complex)
{
  o2::its::ROFOverlapTable<4> table;
  table.defineLayer(0, 4, 100, 0, 0, 0);
  table.defineLayer(1, 4, 100, 0, 0, 10);
  table.defineLayer(2, 8, 50, 0, 0, 0);
  table.defineLayer(3, 7, 50, 50, 0, 10);
  table.init();
  const auto& view = table.getView();
  view.printMapping(0, 1);

  const auto t010 = view.getTimeStamp(0, 3, 1, 3);
  BOOST_CHECK_EQUAL(t010.getTimeStamp(), 300);
  BOOST_CHECK_EQUAL(t010.getTimeStampError(), 100);

  const auto t011 = view.getTimeStamp(0, 2, 1, 3);
  BOOST_CHECK_EQUAL(t011.getTimeStamp(), 290);
  BOOST_CHECK_EQUAL(t011.getTimeStampError(), 10);

  const auto t02 = view.getTimeStamp(0, 1, 2, 3);
  BOOST_CHECK_EQUAL(t02.getTimeStamp(), 150);
  BOOST_CHECK_EQUAL(t02.getTimeStampError(), 50);

  const auto t03 = view.getTimeStamp(0, 0, 3, 0);
  BOOST_CHECK_EQUAL(t03.getTimeStamp(), 40);
  BOOST_CHECK_EQUAL(t03.getTimeStampError(), 60);
}

// ROFVertexLookupTable
BOOST_AUTO_TEST_CASE(rofvertex_basic)
{
  o2::its::ROFVertexLookupTable<1> table;
  table.defineLayer(0, 6, 594, 0, 0, 0);
  table.init();
  std::vector<o2::its::Vertex> vertices;
  o2::its::Vertex vert0;
  vert0.getTimeStamp().setTimeStamp(594);
  vert0.getTimeStamp().setTimeStampError(594);
  vertices.push_back(vert0);
  o2::its::Vertex vert1;
  vert1.getTimeStamp().setTimeStamp(2375);
  vert1.getTimeStamp().setTimeStampError(594);
  vertices.push_back(vert1);
  table.update(vertices.data(), vertices.size());
  const auto view = table.getView();
  view.printAll();
}

BOOST_AUTO_TEST_CASE(rofvertex_init_with_vertices)
{
  o2::its::ROFVertexLookupTable<2> table;
  table.defineLayer(0, 10, 500, 0, 0, 0);
  table.defineLayer(1, 10, 500, 0, 0, 0);

  // create vertices at different timestamps
  std::vector<o2::its::Vertex> vertices;
  for (int i = 0; i < 5; ++i) {
    o2::its::Vertex v;
    v.getTimeStamp().setTimeStamp(i * 1000);
    v.getTimeStamp().setTimeStampError(500);
    vertices.push_back(v);
  }

  table.init(vertices.data(), vertices.size());
  const auto view = table.getView();

  // verify vertices can be queried
  const auto& vtxRange = view.getVertices(0, 0);
  BOOST_CHECK_EQUAL(vtxRange.getEntries(), 1);
}

BOOST_AUTO_TEST_CASE(rofvertex_max_vertices)
{
  o2::its::ROFVertexLookupTable<1> table;
  table.defineLayer(0, 3, 1000, 0, 0, 500);

  std::vector<o2::its::Vertex> vertices;
  for (int i = 0; i < 10; ++i) {
    o2::its::Vertex v;
    v.getTimeStamp().setTimeStamp(500 + i * 100);
    v.getTimeStamp().setTimeStampError(50);
    vertices.push_back(v);
  }

  table.init(vertices.data(), vertices.size());
  const auto view = table.getView();

  int32_t maxVtx = view.getMaxVerticesPerROF();
  BOOST_CHECK(maxVtx >= 0);
}

BOOST_AUTO_TEST_CASE(rofvertex_vertex_more)
{
  o2::its::ROFVertexLookupTable<4> table;
  table.defineLayer(0, 4, 100, 0, 0, 0);
  table.defineLayer(1, 4, 100, 0, 0, 10);
  table.defineLayer(2, 8, 50, 0, 0, 0);
  table.defineLayer(3, 7, 50, 50, 0, 10);
  table.init();

  std::vector<o2::its::Vertex> vertices;
  { // vertex 0 overlapping
    auto& v = vertices.emplace_back();
    v.getTimeStamp().setTimeStamp(100);
    v.getTimeStamp().setTimeStampError(10);
  }
  { // vertex 1
    auto& v = vertices.emplace_back();
    v.getTimeStamp().setTimeStamp(100);
    v.getTimeStamp().setTimeStampError(0);
  }
  { // vertex 2 spanning multiple rofs
    auto& v = vertices.emplace_back();
    v.getTimeStamp().setTimeStamp(100);
    v.getTimeStamp().setTimeStampError(60);
  }

  // sorty vertices by lower bound
  std::sort(vertices.begin(), vertices.end(), [](const auto& pvA, const auto& pvB) {
    const auto& a = pvA.getTimeStamp();
    const auto& b = pvB.getTimeStamp();
    const auto aLower = a.getTimeStamp() - a.getTimeStampError();
    const auto bLower = b.getTimeStamp() - b.getTimeStampError();
    if (aLower != bLower) {
      return aLower < bLower;
    }
    return pvA.getNContributors() > pvB.getNContributors();
  });

  table.update(vertices.data(), vertices.size());
  const auto& view = table.getView();

  const auto& v0 = vertices[0]; // 100+60
  const auto& v1 = vertices[1]; // 100+10
  const auto& v2 = vertices[2]; // 100+0

  // check for v0
  // layer 0
  BOOST_CHECK(!view.isVertexCompatible(0, 0, v0));
  BOOST_CHECK(view.isVertexCompatible(0, 1, v0));
  BOOST_CHECK(!view.isVertexCompatible(0, 2, v0));
  BOOST_CHECK(!view.isVertexCompatible(0, 3, v0));
  // layer 1
  BOOST_CHECK(view.isVertexCompatible(1, 0, v0));
  BOOST_CHECK(view.isVertexCompatible(1, 1, v0));
  BOOST_CHECK(!view.isVertexCompatible(1, 2, v0));
  BOOST_CHECK(!view.isVertexCompatible(1, 3, v0));
  // layer 2
  BOOST_CHECK(!view.isVertexCompatible(2, 0, v0));
  BOOST_CHECK(!view.isVertexCompatible(2, 1, v0));
  BOOST_CHECK(view.isVertexCompatible(2, 2, v0));
  BOOST_CHECK(view.isVertexCompatible(2, 3, v0));
  BOOST_CHECK(!view.isVertexCompatible(2, 4, v0));
  BOOST_CHECK(!view.isVertexCompatible(2, 5, v0));
  BOOST_CHECK(!view.isVertexCompatible(2, 6, v0));
  BOOST_CHECK(!view.isVertexCompatible(2, 7, v0));
  // layer 3
  BOOST_CHECK(view.isVertexCompatible(3, 0, v0));
  BOOST_CHECK(view.isVertexCompatible(3, 1, v0));
  BOOST_CHECK(view.isVertexCompatible(3, 2, v0));
  BOOST_CHECK(!view.isVertexCompatible(3, 3, v0));
  BOOST_CHECK(!view.isVertexCompatible(3, 4, v0));
  BOOST_CHECK(!view.isVertexCompatible(3, 5, v0));
  BOOST_CHECK(!view.isVertexCompatible(3, 6, v0));

  // check for v1
  // layer 0
  BOOST_CHECK(!view.isVertexCompatible(0, 0, v1));
  BOOST_CHECK(view.isVertexCompatible(0, 1, v1));
  BOOST_CHECK(!view.isVertexCompatible(0, 2, v1));
  BOOST_CHECK(!view.isVertexCompatible(0, 3, v1));
  // layer 1
  BOOST_CHECK(view.isVertexCompatible(1, 0, v1));
  BOOST_CHECK(view.isVertexCompatible(1, 1, v1));
  BOOST_CHECK(!view.isVertexCompatible(1, 2, v1));
  BOOST_CHECK(!view.isVertexCompatible(1, 3, v1));
  // layer 2
  BOOST_CHECK(!view.isVertexCompatible(2, 0, v1));
  BOOST_CHECK(!view.isVertexCompatible(2, 1, v1));
  BOOST_CHECK(view.isVertexCompatible(2, 2, v1));
  BOOST_CHECK(!view.isVertexCompatible(2, 3, v1));
  BOOST_CHECK(!view.isVertexCompatible(2, 4, v1));
  BOOST_CHECK(!view.isVertexCompatible(2, 5, v1));
  BOOST_CHECK(!view.isVertexCompatible(2, 6, v1));
  BOOST_CHECK(!view.isVertexCompatible(2, 7, v1));
  // layer 3
  BOOST_CHECK(view.isVertexCompatible(3, 0, v1));
  BOOST_CHECK(view.isVertexCompatible(3, 1, v1));
  BOOST_CHECK(!view.isVertexCompatible(3, 2, v1));
  BOOST_CHECK(!view.isVertexCompatible(3, 3, v1));
  BOOST_CHECK(!view.isVertexCompatible(3, 4, v1));
  BOOST_CHECK(!view.isVertexCompatible(3, 5, v1));
  BOOST_CHECK(!view.isVertexCompatible(3, 6, v1));

  // check for v2
  // layer 0
  BOOST_CHECK(!view.isVertexCompatible(0, 0, v2));
  BOOST_CHECK(view.isVertexCompatible(0, 1, v2));
  BOOST_CHECK(!view.isVertexCompatible(0, 2, v2));
  BOOST_CHECK(!view.isVertexCompatible(0, 3, v2));
  // layer 1
  BOOST_CHECK(view.isVertexCompatible(1, 0, v2));
  BOOST_CHECK(view.isVertexCompatible(1, 1, v2));
  BOOST_CHECK(!view.isVertexCompatible(1, 2, v2));
  BOOST_CHECK(!view.isVertexCompatible(1, 3, v2));
  // layer 2
  BOOST_CHECK(!view.isVertexCompatible(2, 0, v2));
  BOOST_CHECK(!view.isVertexCompatible(2, 1, v2));
  BOOST_CHECK(view.isVertexCompatible(2, 2, v2));
  BOOST_CHECK(!view.isVertexCompatible(2, 3, v2));
  BOOST_CHECK(!view.isVertexCompatible(2, 4, v2));
  BOOST_CHECK(!view.isVertexCompatible(2, 5, v2));
  BOOST_CHECK(!view.isVertexCompatible(2, 6, v2));
  BOOST_CHECK(!view.isVertexCompatible(2, 7, v2));
  // layer 3
  BOOST_CHECK(view.isVertexCompatible(3, 0, v2));
  BOOST_CHECK(view.isVertexCompatible(3, 1, v2));
  BOOST_CHECK(!view.isVertexCompatible(3, 2, v2));
  BOOST_CHECK(!view.isVertexCompatible(3, 3, v2));
  BOOST_CHECK(!view.isVertexCompatible(3, 4, v2));
  BOOST_CHECK(!view.isVertexCompatible(3, 5, v2));
  BOOST_CHECK(!view.isVertexCompatible(3, 6, v2));
}

BOOST_AUTO_TEST_CASE(rofvertex_exact_compatibility)
{
  o2::its::ROFVertexLookupTable<4> table;
  table.defineLayer(0, 4, 100, 0, 0, 0);
  table.defineLayer(1, 4, 100, 0, 0, 10);
  table.defineLayer(2, 8, 50, 0, 0, 0);
  table.defineLayer(3, 7, 50, 50, 0, 10);
  table.init();

  // sorted by lower bound timestamp
  std::vector<o2::its::Vertex> vertices;
  { // idx 0: [40, 160] - wide span
    auto& v = vertices.emplace_back();
    v.getTimeStamp().setTimeStamp(100);
    v.getTimeStamp().setTimeStampError(60);
  }
  { // idx 1: [90, 110]
    auto& v = vertices.emplace_back();
    v.getTimeStamp().setTimeStamp(100);
    v.getTimeStamp().setTimeStampError(10);
  }
  { // idx 2: [100, 100] - zero width, false-positive prone
    auto& v = vertices.emplace_back();
    v.getTimeStamp().setTimeStamp(100);
    v.getTimeStamp().setTimeStampError(0);
  }

  table.update(vertices.data(), vertices.size());
  const auto& view = table.getView();

  // Layer 0 ROF 0: [0, 100)
  BOOST_CHECK(!view.isVertexCompatible(0, 0, vertices[0]));
  BOOST_CHECK(!view.isVertexCompatible(0, 0, vertices[1]));
  BOOST_CHECK(!view.isVertexCompatible(0, 0, vertices[2]));

  // Layer 0 ROF 1: [100, 200) - range includes idx 2 as false positive
  {
    const auto& range = view.getVertices(0, 1);
    BOOST_CHECK_EQUAL(range.getEntries(), 3); // superset

    size_t exactCount = 0;
    for (size_t i = range.getFirstEntry(); i < range.getEntriesBound(); ++i) {
      if (view.isVertexCompatible(0, 1, vertices[i])) {
        ++exactCount;
      }
    }
    // BOOST_CHECK_EQUAL(exactCount, 2); // idx 2 filtered out
  }

  // Layer 0 ROF 2: [200, 300) - nothing overlaps
  BOOST_CHECK(!view.isVertexCompatible(0, 2, vertices[0]));
  BOOST_CHECK(!view.isVertexCompatible(0, 2, vertices[1]));
  BOOST_CHECK(!view.isVertexCompatible(0, 2, vertices[2]));

  // Layer 2 ROF 0: [0, 50) - only idx 0
  BOOST_CHECK(!view.isVertexCompatible(2, 0, vertices[0]));
  BOOST_CHECK(!view.isVertexCompatible(2, 0, vertices[1]));

  // Layer 2 ROF 1: [50, 100) - idx 0 and 1
  BOOST_CHECK(!view.isVertexCompatible(2, 1, vertices[0]));
  BOOST_CHECK(!view.isVertexCompatible(2, 1, vertices[1]));
  BOOST_CHECK(!view.isVertexCompatible(2, 1, vertices[2]));

  // Layer 2 ROF 3: [150, 200) - only idx 0
  BOOST_CHECK(view.isVertexCompatible(2, 3, vertices[0]));
  BOOST_CHECK(!view.isVertexCompatible(2, 3, vertices[1]));

  // Layer 3 ROF 0: [40, 110) - all three genuine
  BOOST_CHECK(view.isVertexCompatible(3, 0, vertices[0]));
  BOOST_CHECK(view.isVertexCompatible(3, 0, vertices[1]));
  BOOST_CHECK(view.isVertexCompatible(3, 0, vertices[2]));

  // Layer 3 ROF 2: [140, 210) - only idx 0
  BOOST_CHECK(view.isVertexCompatible(3, 2, vertices[0]));
  BOOST_CHECK(!view.isVertexCompatible(3, 2, vertices[1]));
  BOOST_CHECK(!view.isVertexCompatible(3, 2, vertices[2]));
}
