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

///
/// \file TrkTGeoBlueprintBuilder.cxx
/// \brief Minimal, material-free TRK-only geometry builder using O2's live TGeo
///

#include "ALICE3ACTS/TrkTGeoBlueprintBuilder.h"

#include "Acts/Definitions/Units.hpp"
#include "Acts/Geometry/Blueprint.hpp"
#include "Acts/Geometry/BlueprintOptions.hpp"
#include "Acts/Geometry/ContainerBlueprintNode.hpp"
#include "Acts/Geometry/CylinderVolumeBounds.hpp"
#include "Acts/Geometry/GeometryIdentifierBlueprintNode.hpp"
#include "Acts/Geometry/LayerBlueprintNode.hpp"
#include "Acts/Geometry/TrackingVolume.hpp"
#include "Acts/Geometry/VolumeAttachmentStrategy.hpp"
#include "Acts/Geometry/VolumeResizeStrategy.hpp"
#include "Acts/Surfaces/CylinderBounds.hpp"
#include "Acts/Surfaces/Surface.hpp"
#include "ActsPlugins/Root/TGeoAxes.hpp"
#include "ActsPlugins/Root/TGeoDetectorElement.hpp"
#include "ActsPlugins/Root/TGeoLayerBuilder.hpp"
#include "ActsPlugins/Root/TGeoParser.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <memory>
#include <string>
#include <ranges>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

#include "TGeoManager.h"
#include "TGeoMatrix.h"
#include "TGeoNode.h"
#include "TGeoVolume.h"

namespace o2::alice3
{
using namespace Acts;
using namespace Acts::Experimental;
using namespace Acts::UnitLiterals;

namespace
{
constexpr double kTGeoToActs = 10.; // cm -> mm
constexpr double kVertexLayerTolerance = 5. * 1_mm;
constexpr double kBarrelLayerTolerance = 15. * 1_mm;
constexpr double kLayerEnvelopeWidth = 2. * 1_mm;

using ElementPtr = std::shared_ptr<ActsPlugins::TGeoDetectorElement>;
using KeyedSurface = std::pair<double, std::shared_ptr<Surface>>;

struct LayerGroup {
  double radius = 0.;
  std::vector<std::shared_ptr<Surface>> surfaces;
};

std::vector<LayerGroup> clusterLayers(std::vector<KeyedSurface> elements, double tolerance)
{
  std::vector<LayerGroup> layers;
  if (elements.empty()) {
    return layers;
  }

  std::ranges::sort(elements, {}, &KeyedSurface::first);
  LayerGroup group;
  double previous = elements.front().first;
  double sum = 0.;

  for (const auto& [radius, surface] : elements) {
    if (!group.surfaces.empty() && radius - previous > tolerance) {
      group.radius = sum / group.surfaces.size();
      layers.push_back(std::move(group));
      group = LayerGroup{};
      sum = 0.;
    }
    group.surfaces.push_back(surface);
    sum += radius;
    previous = radius;
  }
  group.radius = sum / group.surfaces.size();
  layers.push_back(std::move(group));
  return layers;
}

} // namespace

TrkTGeoBlueprintBuilder::TrkTGeoBlueprintBuilder() : TrkTGeoBlueprintBuilder(Config{}) {}

TrkTGeoBlueprintBuilder::TrkTGeoBlueprintBuilder(Config config) : mConfig(std::move(config)) {}

o2::acts::TrackingGeometryOutput TrkTGeoBlueprintBuilder::build(TGeoManager& tgeo,
                                                                const Acts::GeometryContext& gctx)
{
  ACTS_LOCAL_LOGGER(getDefaultLogger("ALICE3TRKAuto", mConfig.logLevel));
  ACTS_INFO("Building ACTS TRK geometry from the live TGeo '" << tgeo.GetName() << "'");

  auto* topVolume = tgeo.GetTopVolume();
  if (!topVolume) {
    throw std::runtime_error("ALICE3 TRK: no top volume in the loaded TGeo geometry");
  }

  // TGeoParser selects the placed TRK sensitive nodes, including the VD
  // PETALCASE surfaces. Avoid parsing every support/passive simulation volume.
  ActsPlugins::TGeoParser::Options options;
  options.volumeNames = {topVolume->GetName()};
  options.targetNames = {"*TRKSensor*"};
  options.unit = kTGeoToActs;

  ActsPlugins::TGeoParser::State state;
  state.volume = topVolume;
  state.onBranch = true;
  TGeoHMatrix transform = TGeoIdentity(topVolume->GetName());
  ActsPlugins::TGeoParser::select(state, options, transform);

  if (state.selectedNodes.empty()) {
    throw std::runtime_error("ALICE3 TRK: no TRK sensitive TGeo nodes found (*TRKSensor*)");
  }

  const auto thinY = ActsPlugins::TGeoAxes::parse("ZXY");
  const auto thinZ = ActsPlugins::TGeoAxes::parse("XYZ");
  std::vector<ElementPtr> elements;
  std::vector<KeyedSurface> vertexSurfaces;
  std::vector<KeyedSurface> barrelSurfaces;
  elements.reserve(state.selectedNodes.size());

  for (const auto& snode : state.selectedNodes) {
    const std::string_view name = snode.node->GetVolume()->GetName();
    const bool vertex = name.find("PETALCASE") != std::string_view::npos;
    const auto axes = vertex ? thinZ : thinY;

    auto element = ActsPlugins::TGeoLayerBuilder::defaultElementFactory(
      ActsPlugins::TGeoDetectorElement::Identifier{}, *snode.node,
      *snode.transform, axes, kTGeoToActs, nullptr);
    auto surface = element->surface().getSharedPtr();

    if (vertex && surface->type() != Surface::SurfaceType::Cylinder) {
      throw std::runtime_error("ALICE3 TRK: expected a cylindrical vertex-detector sensor: " + std::string(name));
    }
    // TRK is a barrel detector. A misidentified forward surface must not
    // silently enter a cylindrical layer and break ACTS navigation.
    const auto center = surface->center(gctx);
    const double radius = (surface->type() == Surface::SurfaceType::Cylinder)
                            ? surface->bounds().values()[CylinderBounds::eR]
                            : std::hypot(center.x(), center.y());
    if (!vertex) {
      const auto normal = surface->normal(gctx, center, Vector3::UnitZ());
      if (std::abs(normal.z()) > 0.7) {
        throw std::runtime_error("ALICE3 TRK: a non-barrel surface matched *TRKSensor*: " + std::string(name));
      }
    }

    (vertex ? vertexSurfaces : barrelSurfaces).emplace_back(radius, std::move(surface));
    elements.push_back(std::move(element));
  }

  auto layers = clusterLayers(std::move(vertexSurfaces), kVertexLayerTolerance);
  auto barrelLayers = clusterLayers(std::move(barrelSurfaces), kBarrelLayerTolerance);
  for (auto& layer : barrelLayers) {
    layers.push_back(std::move(layer));
  }
  std::ranges::sort(layers, {}, &LayerGroup::radius);

  if (layers.empty()) {
    throw std::runtime_error("ALICE3 TRK: no barrel layers found in the loaded TGeo");
  }
  ACTS_INFO("Found " << elements.size() << " sensors in " << layers.size() << " TRK layers");
  for (std::size_t i = 0; i < layers.size(); ++i) {
    ACTS_INFO("  TRK layer " << i << ": R=" << layers[i].radius / 1_mm
                             << " mm, sensors=" << layers[i].surfaces.size());
  }

  Blueprint::Config config;
  // Enlarge the inner world radius to include the beamline (r = 0).
  const double innerEnvelope = layers.front().radius + 20. * 1_mm;
  config.envelope = ExtentEnvelope{{.z = {20. * 1_mm, 20. * 1_mm},
                                    .r = {innerEnvelope, 20. * 1_mm}}};
  Blueprint root{config};
  auto& ids = root.withGeometryIdentifier();
  const auto byRadius = [](const TrackingVolume& a, const TrackingVolume& b) {
    const auto midRadius = [](const TrackingVolume& volume) {
      const auto& bounds = dynamic_cast<const CylinderVolumeBounds&>(volume.volumeBounds());
      return 0.5 * (bounds.get(CylinderVolumeBounds::eMinR) + bounds.get(CylinderVolumeBounds::eMaxR));
    };
    return midRadius(a) < midRadius(b);
  };
  // One self-contained TRK region. IDs are deterministic within this builder,
  // but they are NOT compatible with the full-detector geometry/material map.
  ids.setAllVolumeIdsTo(40).incrementLayerIds(1).sortBy(byRadius);
  auto& barrel = ids.addCylinderContainer("TRKBarrel", AxisDirection::AxisR);
  barrel.setAttachmentStrategy(VolumeAttachmentStrategy::Gap);
  barrel.setResizeStrategies(VolumeResizeStrategy::Gap, VolumeResizeStrategy::Gap);

  const ExtentEnvelope layerEnvelope = ExtentEnvelope{{.z = {kLayerEnvelopeWidth, kLayerEnvelopeWidth},
                                                       .r = {kLayerEnvelopeWidth, kLayerEnvelopeWidth}}};
  for (std::size_t i = 0; i < layers.size(); ++i) {
    barrel.addLayer("TRKBarrel_L" + std::to_string(i), [&layers, &layerEnvelope, i](LayerBlueprintNode& node) {
      node.setSurfaces(layers[i].surfaces);
      node.setLayerType(LayerBlueprintNode::LayerType::Cylinder);
      node.setEnvelope(layerEnvelope);
      node.setUseCenterOfGravity(false, false, true);
    });
  }

  if (!mConfig.graphvizFile.empty()) {
    std::ofstream out(mConfig.graphvizFile);
    if (!out) {
      throw std::runtime_error("ALICE3 TRK: cannot write Graphviz file: " + mConfig.graphvizFile);
    }
    root.graphviz(out);
  }

  BlueprintOptions opts;
  auto trackingGeometry = root.construct(opts, gctx, logger());

  o2::acts::TrackingGeometryOutput output;
  output.elementStore.reserve(elements.size());
  for (const auto& element : elements) {
    output.elementStore.push_back(element);
  }
  output.geometry = std::shared_ptr<Acts::TrackingGeometry>(std::move(trackingGeometry));
  ACTS_INFO("ACTS TRK geometry constructed from the loaded TGeo");
  return output;
}

} // namespace o2::alice3
