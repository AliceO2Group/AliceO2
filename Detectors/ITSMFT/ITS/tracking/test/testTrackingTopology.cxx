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
#define BOOST_TEST_MODULE ITS TrackingTopology
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>
#include "ITStracking/TrackingTopology.h"

/// -------- Tests --------
BOOST_AUTO_TEST_CASE(layermask_holes_and_length)
{
  using o2::its::LayerMask;

  const LayerMask layer3Hole{0x77}; // layers 0,1,2,4,5,6
  BOOST_CHECK_EQUAL(layer3Hole.count(), 6);
  BOOST_CHECK_EQUAL(layer3Hole.length(), 7);
  BOOST_CHECK_EQUAL(layer3Hole.holeMask().value(), 0x08);
  BOOST_CHECK(layer3Hole.isAllowed(1, 0x08));
  BOOST_CHECK(!layer3Hole.isAllowed(0, 0x08));

  const LayerMask missingLeadingLayer0{0x7e}; // layers 1..6
  BOOST_CHECK_EQUAL(missingLeadingLayer0.count(), 6);
  BOOST_CHECK_EQUAL(missingLeadingLayer0.length(), 6);
  BOOST_CHECK_EQUAL(missingLeadingLayer0.holeMask().value(), 0x00);

  const LayerMask missingTrailingLayer6{0x3f}; // layers 0..5
  BOOST_CHECK_EQUAL(missingTrailingLayer6.count(), 6);
  BOOST_CHECK_EQUAL(missingTrailingLayer6.length(), 6);
  BOOST_CHECK_EQUAL(missingTrailingLayer6.holeMask().value(), 0x00);
}

BOOST_AUTO_TEST_CASE(layermask_topological_length_counts_internal_holes)
{
  using o2::its::LayerMask;

  BOOST_CHECK_GE(LayerMask{0x7f}.length(), 7); // 7 clusters
  BOOST_CHECK_GE(LayerMask{0x77}.length(), 7); // 6 clusters + layer-3 hole
  BOOST_CHECK_LT(LayerMask{0x7e}.length(), 7); // missing leading layer
  BOOST_CHECK_LT(LayerMask{0x3f}.length(), 7); // missing trailing layer
}

BOOST_AUTO_TEST_CASE(trackingtopology_basic)
{
  o2::its::TrackingTopology<4> topo;
  topo.init(4, 0, 0);
  const auto view = topo.getView();
  view.print();

  BOOST_CHECK_EQUAL(view.nLinks, 3);
  for (int i{0}; i < 3; ++i) {
    const auto& tra = view.getLink(i);
    BOOST_CHECK_EQUAL(tra.fromLayer, i);
    BOOST_CHECK_EQUAL(tra.toLayer, i + 1);
  }

  BOOST_CHECK_EQUAL(view.nCells, 2);
  for (int i{0}; i < 2; ++i) {
    const auto& cell = view.getCell(i);
    BOOST_CHECK_EQUAL(cell.firstLink, i);
    BOOST_CHECK_EQUAL(cell.secondLink, i + 1);
  }
}

/// Without holes the cell graph is a single chain, so cell i - spanning layers i, i+1, i+2 -
/// can only ever be reached by the i cells below it.
BOOST_AUTO_TEST_CASE(trackingtopology_max_cell_level_is_the_chain_depth)
{
  o2::its::TrackingTopology<7> topo;
  topo.init(7, 0, 0);
  const auto view = topo.getView();
  view.print();

  BOOST_REQUIRE_EQUAL(view.nLinks, 6);
  BOOST_REQUIRE_EQUAL(view.nCells, 5);
  for (int i{0}; i < view.nCells; ++i) {
    BOOST_CHECK_EQUAL(int(view.getMaxCellLevel(i)), i + 1);
  }
}

/// With a hole allowed the graph branches, and the depth is the longest path ending on a cell
/// rather than its index. Every cell must still be reachable by at least one chain, and no cell
/// may claim a level deeper than the number of cells that could precede it.
BOOST_AUTO_TEST_CASE(trackingtopology_max_cell_level_follows_the_longest_path)
{
  o2::its::TrackingTopology<5> topo;
  topo.init(5, 1, 1 << 2);
  const auto view = topo.getView();
  view.print();

  bool sawBranching = false;
  for (int i{0}; i < view.nCells; ++i) {
    const auto level = int(view.getMaxCellLevel(i));
    BOOST_CHECK_GE(level, 1);
    BOOST_CHECK_LE(level, int(view.nCells));
    // A cell reached by a chain of n predecessors needs n+2 layers below its outer one.
    BOOST_CHECK_LE(level, view.getCell(i).hitLayerMask.last() - 1);
    sawBranching |= level != i + 1;
  }
  BOOST_CHECK(sawBranching); // otherwise this is just the chain case again
}

/// Neighbour construction can finalize a target after one source-layer wave: every predecessor
/// of a target ends on the destination layer of the target's first link.
BOOST_AUTO_TEST_CASE(trackingtopology_predecessors_belong_to_one_layer_wave)
{
  o2::its::TrackingTopology<7> topo;
  topo.init(7, 2, (1 << 2) | (1 << 4));
  const auto view = topo.getView();

  for (int sourceId{0}; sourceId < view.nCells; ++sourceId) {
    const auto& source = view.getCell(sourceId);
    const int sourceWave = source.hitLayerMask.last();
    const auto successors = view.getCellsStartingWithLink(source.secondLink);
    for (int i{0}; i < successors.getEntries(); ++i) {
      const int targetId = view.cellsByFirstLink[successors.getFirstEntry() + i];
      const auto& target = view.getCell(targetId);
      BOOST_CHECK_EQUAL(target.firstLink, source.secondLink);
      BOOST_CHECK_EQUAL(sourceWave, view.getLink(target.firstLink).toLayer);
    }
  }
}

BOOST_AUTO_TEST_CASE(trackingtopology_single_allowed_hole)
{
  o2::its::TrackingTopology<5> topo;
  topo.init(5, 1, 1 << 2);
  const auto view = topo.getView();
  view.print();

  BOOST_CHECK_EQUAL(view.nLinks, 5);
  BOOST_CHECK_EQUAL(view.nCells, 5);

  bool hasHoleLink = false;
  for (int i{0}; i < view.nLinks; ++i) {
    const auto& link = view.getLink(i);
    hasHoleLink |= link.fromLayer == 1 && link.toLayer == 3;
    BOOST_CHECK(o2::its::LayerMask::skipped(link.fromLayer, link.toLayer).isAllowedHoleMask(1, 1 << 2));
  }
  BOOST_CHECK(hasHoleLink);

  bool hasHoleCell = false;
  for (int i{0}; i < view.nCells; ++i) {
    const auto& cell = view.getCell(i);
    hasHoleCell |= cell.hitLayerMask.value() == 0x0b; // layers 0,1,3
    BOOST_CHECK(cell.hitLayerMask.isAllowed(1, 1 << 2));
  }
  BOOST_CHECK(hasHoleCell);
}

BOOST_AUTO_TEST_CASE(trackingtopology_rejects_wrong_hole_layer)
{
  o2::its::TrackingTopology<5> topo;
  topo.init(5, 1, 1 << 2);
  const auto view = topo.getView();
  view.print();

  for (int i{0}; i < view.nLinks; ++i) {
    const auto& link = view.getLink(i);
    BOOST_CHECK(!(link.fromLayer == 0 && link.toLayer == 2));
    BOOST_CHECK(!(link.fromLayer == 2 && link.toLayer == 4));
  }

  for (int i{0}; i < view.nCells; ++i) {
    const auto& cell = view.getCell(i);
    BOOST_CHECK(cell.hitLayerMask.holeMask().isSubsetOf(1 << 2));
  }
}
