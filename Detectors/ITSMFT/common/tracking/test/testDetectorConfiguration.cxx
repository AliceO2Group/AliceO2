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

#define BOOST_TEST_MODULE ITSMFT DetectorConfiguration
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>

#include <algorithm>
#include <bit>
#include <vector>

#include "ITStracking/Configuration.h"
#include "MFTTracking/Constants.h"
#include "ITSMFTTracking/Configuration.h"
#include "ITSMFTTracking/TraversalTopology.h"

namespace
{
using namespace o2::itsmft::tracking;
using o2::itsmft::TrackingParameters;

std::vector<SurfaceDescriptor> catalog(uint16_t count, SurfaceKind kind = SurfaceKind::Cylinder)
{
  std::vector<SurfaceDescriptor> result;
  for (uint16_t id = 0; id < count; ++id) {
    result.emplace_back(id, 0, kind);
  }
  return result;
}

LayerMask mask(std::initializer_list<uint16_t> ids)
{
  LayerMask result;
  for (const auto id : ids) {
    result.set(id);
  }
  return result;
}

TrackingParameters parametersFor(const DetectorConfiguration& layout)
{
  TrackingParameters parameters;
  parameters.NLayers = static_cast<int>(layout.size());
  parameters.StartLayerMask = LayerMask::span(0, parameters.NLayers - 1);
  return parameters;
}
} // namespace

BOOST_AUTO_TEST_CASE(LayerMaskCoversThirtyTwoLayoutPositions)
{
  LayerMask surfaces;
  surfaces.set(0);
  surfaces.set(16);
  surfaces.set(31);
  BOOST_CHECK(surfaces.has(0));
  BOOST_CHECK(surfaces.has(16));
  BOOST_CHECK(surfaces.has(31));
  BOOST_CHECK_EQUAL(surfaces.count(), 3);
}

BOOST_AUTO_TEST_CASE(LayoutValidatesLimitsAndDerivesDenseIds)
{
  const auto surfaces = catalog(33);
  const auto layout = DetectorConfiguration{surfaces};
  BOOST_CHECK(layout.getError() == DetectorConfigurationError::TooManySurfaces);

  const auto dense = catalog(4);
  const auto valid = DetectorConfiguration{dense};
  BOOST_CHECK(valid.valid());
  BOOST_CHECK_EQUAL(valid.size(), 4u);
  for (uint16_t position = 0; position < valid.size(); ++position) {
    BOOST_CHECK(&valid[LayerId{position}] == &valid.getLayers()[position]);
  }
}

BOOST_AUTO_TEST_CASE(ComponentBoundariesAndKindIndependentCatalogs)
{
  const auto mixed = std::vector<SurfaceDescriptor>{{0, 0, SurfaceKind::Cylinder},
                                                    {1, 0, SurfaceKind::Cylinder},
                                                    {0, 8, SurfaceKind::Disk},
                                                    {1, 8, SurfaceKind::Disk}};
  const std::vector<uint16_t> componentOffsets = {0, 2};
  const auto layout = DetectorConfiguration{mixed, componentOffsets};
  BOOST_REQUIRE(layout.valid());
  BOOST_CHECK(layout.sameComponent(0, 1));
  BOOST_CHECK(!layout.sameComponent(1, 2));

  const auto topology = deriveTraversalTopology(layout, parametersFor(layout));
  BOOST_REQUIRE(topology.ok());
  BOOST_CHECK_EQUAL(topology.topology->edges.size(), 2u);
  BOOST_CHECK(std::all_of(topology.topology->edges.begin(), topology.topology->edges.end(), [](const Edge& edge) {
    return edge.from.value() / 2 == edge.to.value() / 2;
  }));
}

BOOST_AUTO_TEST_CASE(HoleAndSeedPoliciesProduceSparseTopology)
{
  const std::vector<SurfaceDescriptor> surfaces = catalog(4);
  const auto layout = DetectorConfiguration{surfaces, {0}, mask({1})};
  auto parameters = parametersFor(layout);
  parameters.MaxHoles = 1;
  parameters.StartLayerMask = LayerMask{1u << 3};
  parameters.InactiveLayerMask = LayerMask{1u << 1};
  const auto result = deriveTraversalTopology(layout, parameters);
  BOOST_REQUIRE(result.ok());
  const auto& topology = *result.topology;
  BOOST_CHECK_EQUAL(topology.activeSurfaceList.size(), 3u);
  BOOST_CHECK_EQUAL(topology.nLayers, 4u);
  BOOST_CHECK(topology.activeSurfaceList[1] == LayerId{2});
  BOOST_CHECK_EQUAL(topology.edges.size(), 2u);
  BOOST_CHECK_EQUAL(topology.paths.size(), 1u);
  BOOST_CHECK(topology.edges[0].from == LayerId{0});
  BOOST_CHECK(topology.edges[0].to == LayerId{2});
  BOOST_REQUIRE_EQUAL(topology.roadStartPaths.size(), 1u);
  BOOST_CHECK(topology.getView(layout.getSurfaceCatalog()).getPath(topology.roadStartPaths.front()).first == EdgeId{0});
}

BOOST_AUTO_TEST_CASE(InvalidLayoutAndLayerCountDerivationIsTransactional)
{
  const auto surfaces = catalog(4);
  const auto layout = DetectorConfiguration{surfaces};
  auto wrongLayerCount = parametersFor(layout);
  wrongLayerCount.NLayers = 7;
  const auto invalidCount = deriveTraversalTopology(layout, wrongLayerCount);
  BOOST_CHECK(!invalidCount.ok());
  BOOST_CHECK(!invalidCount.topology.has_value());
  BOOST_CHECK(invalidCount.error == TraversalTopologyError::LayerCountMismatch);

  const auto invalidLayout = deriveTraversalTopology(DetectorConfiguration{}, TrackingParameters{});
  BOOST_CHECK(!invalidLayout.ok());
  BOOST_CHECK(!invalidLayout.topology.has_value());
}

BOOST_AUTO_TEST_CASE(RepresentativeRadiiMatchProductionDefaultsBitExactly)
{
  const DetectorConfiguration its{kITSSurfaces};
  const o2::its::TrackingParameters productionITS;
  for (uint16_t layer = 0; layer < ITSNLayers; ++layer) {
    BOOST_CHECK_EQUAL(std::bit_cast<uint32_t>(its.getRepresentativeRadius(LayerId{layer})),
                      std::bit_cast<uint32_t>(productionITS.LayerRadii[layer]));
  }

  const DetectorConfiguration mft{kMFTSurfaces};
  for (uint16_t layer = 0; layer < MFTNLayers; ++layer) {
    const float productionRadius = 0.5f * (o2::mft::constants::index_table::RMin[layer] +
                                           o2::mft::constants::index_table::RMax[layer]);
    BOOST_CHECK_EQUAL(std::bit_cast<uint32_t>(mft.getRepresentativeRadius(LayerId{layer})),
                      std::bit_cast<uint32_t>(productionRadius));
  }
}

BOOST_AUTO_TEST_CASE(RepresentativeRadiusFollowsGeometryInMixedConfigurations)
{
  auto surfaces = std::vector<SurfaceDescriptor>{kMFTSurfaces[2], kITSSurfaces[4]};
  surfaces[0].chartRange = {4.f, 12.f};
  surfaces[0].referenceCoordinate = -100.f;
  surfaces[1].referenceCoordinate = 42.f;
  surfaces[1].chartRange = {-30.f, 30.f};
  const DetectorConfiguration detector{surfaces, {0, 1}};
  BOOST_REQUIRE(detector.valid());
  BOOST_CHECK_EQUAL(detector.getRepresentativeRadius(LayerId{0}), 8.f);
  BOOST_CHECK_EQUAL(detector.getRepresentativeRadius(LayerId{1}), 42.f);
}
