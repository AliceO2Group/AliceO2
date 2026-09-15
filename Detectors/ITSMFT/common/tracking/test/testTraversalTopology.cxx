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

#define BOOST_TEST_MODULE ITSMFT TraversalTopology
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>

#include <algorithm>
#include <type_traits>
#include <vector>

#include "ITSMFTTracking/Configuration.h"
#include "ITSMFTTracking/TraversalTopology.h"

namespace
{
using namespace o2::itsmft::tracking;
using o2::itsmft::TrackingParameters;

std::vector<SurfaceDescriptor> catalog(uint16_t count)
{
  std::vector<SurfaceDescriptor> result;
  result.reserve(count);
  for (uint16_t id = 0; id < count; ++id) {
    result.push_back(SurfaceDescriptor{id, 0, SurfaceKind::Cylinder});
  }
  return result;
}

DetectorLayout makeLayout(uint16_t layerCount,
                          std::vector<uint16_t> componentOffsets = {0},
                          LayerMask holeLayers = {})
{
  const auto surfaces = catalog(layerCount);
  DetectorLayoutDefinition definition;
  definition.componentOffsets = std::move(componentOffsets);
  definition.holeLayers = holeLayers;
  return DetectorLayout{surfaces, std::move(definition)};
}

LayerMask mask(std::initializer_list<uint16_t> ids)
{
  LayerMask result;
  for (const auto id : ids) {
    result.set(id);
  }
  return result;
}

LayerMask layerMask(std::initializer_list<uint16_t> positions)
{
  LayerMask result;
  for (const auto position : positions) {
    result.set(position);
  }
  return result;
}

TrackingParameters parametersFor(const DetectorLayout& layout)
{
  TrackingParameters result;
  result.NLayers = static_cast<int>(layout.size());
  result.StartLayerMask = LayerMask::span(0, result.NLayers - 1);
  return result;
}

const Edge* findEdge(const TraversalTopology& topology, LayerId from, LayerId to)
{
  const auto edge = std::find_if(topology.edges.begin(), topology.edges.end(), [&](const auto& candidate) {
    return candidate.from == from && candidate.to == to;
  });
  return edge == topology.edges.end() ? nullptr : &*edge;
}
} // namespace

BOOST_AUTO_TEST_CASE(CellPathContainsOnlyTwoEdgeIds)
{
  static_assert(std::is_standard_layout_v<CellPath>);
  static_assert(std::is_trivially_copyable_v<CellPath>);
  static_assert(std::is_same_v<decltype(CellPath::first), EdgeId>);
  static_assert(std::is_same_v<decltype(CellPath::second), EdgeId>);
  static_assert(sizeof(CellPath) == sizeof(EdgeId) + sizeof(EdgeId));
  BOOST_CHECK_EQUAL(sizeof(CellPath), 4u);
}

BOOST_AUTO_TEST_CASE(EdgeContainsOnlySurfaceEndpoints)
{
  static_assert(std::is_standard_layout_v<Edge>);
  static_assert(std::is_trivially_copyable_v<Edge>);
  static_assert(std::is_same_v<decltype(Edge::from), LayerId>);
  static_assert(std::is_same_v<decltype(Edge::to), LayerId>);
  static_assert(sizeof(Edge) == sizeof(LayerId) + sizeof(LayerId));
  BOOST_CHECK_EQUAL(sizeof(Edge), 4u);
}

BOOST_AUTO_TEST_CASE(ComponentBoundariesRejectCrossComponentEdges)
{
  const auto layout = makeLayout(4, {0, 2});
  const auto result = deriveTraversalTopology(layout, parametersFor(layout));
  BOOST_REQUIRE(result.ok());
  BOOST_CHECK_EQUAL(result.topology->edges.size(), 2u);
  BOOST_CHECK(findEdge(*result.topology, LayerId{1}, LayerId{2}) == nullptr);
}

BOOST_AUTO_TEST_CASE(AllActiveChainDerivesEdgesAndCellPaths)
{
  const auto layout = makeLayout(4);
  const auto result = deriveTraversalTopology(layout, parametersFor(layout));
  BOOST_REQUIRE(result.ok());
  const auto& topology = *result.topology;
  BOOST_CHECK_EQUAL(topology.nLayers, 4u);
  BOOST_CHECK_EQUAL(topology.activeSurfaceList.size(), 4u);
  BOOST_CHECK_EQUAL(topology.edges.size(), 3u);
  BOOST_CHECK_EQUAL(topology.paths.size(), 2u);
  BOOST_CHECK(topology.edges[0].from == LayerId{0});
  BOOST_CHECK(topology.edges[0].to == LayerId{1});
  BOOST_CHECK(topology.edges[1].from == LayerId{1});
  BOOST_CHECK(topology.edges[1].to == LayerId{2});
  BOOST_CHECK(topology.edges[2].from == LayerId{2});
  BOOST_CHECK(topology.edges[2].to == LayerId{3});
  BOOST_CHECK(topology.paths[0].first == EdgeId{0});
  BOOST_CHECK(topology.paths[0].second == EdgeId{1});
  BOOST_CHECK(topology.paths[1].first == EdgeId{1});
  BOOST_CHECK(topology.paths[1].second == EdgeId{2});
}

BOOST_AUTO_TEST_CASE(SeedingLayersBuildTheGraphWhileStartLayersOnlySelectRoadStarts)
{
  const auto layout = makeLayout(5);
  const auto seeding = mask({0, 2, 4});
  auto outerStartParameters = parametersFor(layout);
  outerStartParameters.SeedingLayers = layerMask({0, 2, 4});
  outerStartParameters.StartLayerMask = layerMask({4});
  const auto startsAtOuterSurface = deriveTraversalTopology(
    layout, outerStartParameters);
  BOOST_REQUIRE(startsAtOuterSurface.ok());
  const auto& topology = *startsAtOuterSurface.topology;
  BOOST_CHECK(topology.seedingLayers == seeding);
  BOOST_CHECK_EQUAL(topology.activeSurfaceList.size(), 5u);
  BOOST_REQUIRE_EQUAL(topology.edges.size(), 2u);
  BOOST_CHECK(findEdge(topology, LayerId{0}, LayerId{2}) != nullptr);
  BOOST_CHECK(findEdge(topology, LayerId{2}, LayerId{4}) != nullptr);
  BOOST_REQUIRE_EQUAL(topology.paths.size(), 1u);
  BOOST_REQUIRE_EQUAL(topology.roadStartPaths.size(), 1u);

  auto middleStartParameters = outerStartParameters;
  middleStartParameters.StartLayerMask = layerMask({2});
  const auto startsAtMiddleSurface = deriveTraversalTopology(
    layout, middleStartParameters);
  BOOST_REQUIRE(startsAtMiddleSurface.ok());
  BOOST_REQUIRE_EQUAL(startsAtMiddleSurface.topology->edges.size(), topology.edges.size());
  BOOST_REQUIRE_EQUAL(startsAtMiddleSurface.topology->paths.size(), topology.paths.size());
  for (std::size_t i = 0; i < topology.edges.size(); ++i) {
    BOOST_CHECK(startsAtMiddleSurface.topology->edges[i].from == topology.edges[i].from);
    BOOST_CHECK(startsAtMiddleSurface.topology->edges[i].to == topology.edges[i].to);
  }
  for (std::size_t i = 0; i < topology.paths.size(); ++i) {
    BOOST_CHECK(startsAtMiddleSurface.topology->paths[i].first == topology.paths[i].first);
    BOOST_CHECK(startsAtMiddleSurface.topology->paths[i].second == topology.paths[i].second);
  }
  BOOST_CHECK(startsAtMiddleSurface.topology->roadStartPaths.empty());
}

BOOST_AUTO_TEST_CASE(DisabledMiddleSurfaceRetainsAdmittedBridge)
{
  const auto layout = makeLayout(4, {0}, mask({1}));
  auto parameters = parametersFor(layout);
  parameters.MaxHoles = 1;
  parameters.InactiveLayerMask = layerMask({1});
  const auto result = deriveTraversalTopology(layout, parameters);
  BOOST_REQUIRE(result.ok());
  const auto& topology = *result.topology;
  BOOST_CHECK_EQUAL(topology.activeSurfaceList.size(), 3u);
  BOOST_CHECK_EQUAL(topology.edges.size(), 2u);
  BOOST_CHECK_EQUAL(topology.paths.size(), 1u);
  const auto* bridge = findEdge(topology, LayerId{0}, LayerId{2});
  BOOST_REQUIRE(bridge != nullptr);
  BOOST_CHECK(bridge->from == LayerId{0});
  BOOST_CHECK(bridge->to == LayerId{2});
  BOOST_CHECK(topology.activeSurfaceList[1] == LayerId{2});
  BOOST_CHECK(topology.paths[0].first == EdgeId{0});
  BOOST_CHECK(topology.paths[0].second == EdgeId{1});
}

BOOST_AUTO_TEST_CASE(DisabledEndpointOmitsItsEdges)
{
  const auto layout = makeLayout(4, {0}, mask({1}));
  auto parameters = parametersFor(layout);
  parameters.MaxHoles = 1;
  parameters.InactiveLayerMask = layerMask({0});
  const auto result = deriveTraversalTopology(layout, parameters);
  BOOST_REQUIRE(result.ok());
  for (const auto& edge : result.topology->edges) {
    BOOST_CHECK(edge.from != LayerId{0});
    BOOST_CHECK(edge.to != LayerId{0});
  }
  BOOST_CHECK(findEdge(*result.topology, LayerId{1}, LayerId{2}) != nullptr);
}

BOOST_AUTO_TEST_CASE(InvalidDerivationIsTransactional)
{
  const auto layout = makeLayout(4);
  auto parameters = parametersFor(layout);
  parameters.NLayers = 7;
  const auto result = deriveTraversalTopology(layout, parameters);
  BOOST_CHECK(!result.ok());
  BOOST_CHECK(!result.topology.has_value());
  BOOST_CHECK(result.error == TraversalTopologyError::LayerCountMismatch);

  DetectorLayout invalid;
  const auto invalidResult = deriveTraversalTopology(invalid, TrackingParameters{});
  BOOST_CHECK(!invalidResult.ok());
  BOOST_CHECK(!invalidResult.topology.has_value());
  BOOST_CHECK(invalidResult.error == TraversalTopologyError::InvalidLayout);
}
