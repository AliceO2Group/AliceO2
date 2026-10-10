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
/// \file Gen3BlueprintBuilder.cxx
/// \author Paolo Butti
///
/// Ported from actsO2 (ActsAlgorithms/Geometry/src/ALICE3Gen3Geometry.cpp).
///
/// Gen3 ("Blueprint") tracking geometry for ALICE 3. Geometry-specific numbers
/// live in the JSON loaded into Gen3GeometryConfig; this file is construction
/// logic.
///
/// Stacking rule: Gen3 volume stacks only expand (R-stack -> common z halfZ,
/// Z-stack -> common radius), so children must not overlap along the stack axis.
/// A flat [negEC | barrel | posEC] split fails because OTOF (halfZ 3400) would
/// pad the barrel out to |z|<3400 and swallow the FT3 discs. Layers are instead
/// separated radially where they overlap in z, and by z where they overlap in r.
/// Gap volumes between children are inserted automatically.
///
///   Core                        R-stack   (|z| <= 3400)     volume ID
///    |- Main                     Z-stack   (r <~ 860)
///    |   |- FwdNeg               R-stack   FT3 outer discs + fwd shells    20
///    |   |- Central              R-stack   (|z| <= 1283.5)
///    |   |   |- InnerCore        Z-stack   (r <~ 400)
///    |   |   |   |- MiddleDisksNeg     FT3 inner discs                     30
///    |   |   |   |- InnerBarrel   R-stack  vertex + TRK(70..300) + ITOF    40
///    |   |   |   |- MiddleDisksPos     FT3 inner discs                     60
///    |   |   |- OuterTrackerBarrel R-stack TRK 450,600,800                 50
///    |   |- FwdPos               R-stack                                   70
///    |- OTOF                     cylinder  r = 920                         90
///
/// Volume IDs are pinned per region (config volumeIds), layer IDs increment from
/// 1 within each. Layers are discovered by clustering sensor positions, per
/// subsystem (see Subsystem / clusterBy) - staggered TRK rows and FT3 double
/// planes merge into one layer each.
///

#include "ALICE3ACTS/Gen3BlueprintBuilder.h"

// Geometry-specific parameters (radii, z, tolerances, sensor names, axes,
// volume IDs, passive cylinders/discs) are loaded at runtime from a JSON file
// into a Gen3GeometryConfig. Supply a different JSON to target a different
// geometry - no recompilation needed.
#include "ALICE3ACTS/Gen3GeometryConfig.h"

#include "Acts/Definitions/Units.hpp"
#include "Acts/Geometry/Blueprint.hpp"
#include "Acts/Geometry/BlueprintOptions.hpp"
#include "Acts/Geometry/ContainerBlueprintNode.hpp"
#include "Acts/Geometry/GeometryContext.hpp"
#include "Acts/Geometry/CylinderVolumeBounds.hpp"
#include "Acts/Geometry/GeometryIdentifierBlueprintNode.hpp"
#include "Acts/Geometry/LayerBlueprintNode.hpp"
#include "Acts/Geometry/TrackingGeometry.hpp"
#include "Acts/Geometry/TrackingVolume.hpp"
#include "Acts/Material/ProtoSurfaceMaterial.hpp"
#include "Acts/Utilities/BinUtility.hpp"
#include "Acts/Utilities/BinningType.hpp"
#include "Acts/Geometry/VolumeAttachmentStrategy.hpp"
#include "Acts/Geometry/VolumeResizeStrategy.hpp"
#include "Acts/Surfaces/CylinderBounds.hpp"
#include "Acts/Surfaces/CylinderSurface.hpp"
#include "Acts/Surfaces/DiscBounds.hpp"
#include "Acts/Surfaces/DiscSurface.hpp"
#include "Acts/Surfaces/Surface.hpp"
#include "Acts/Utilities/AxisDefinitions.hpp"
#include "Acts/Utilities/ProtoAxis.hpp"
#include "ActsPlugins/Root/TGeoAxes.hpp"
#include "ActsPlugins/Root/TGeoDetectorElement.hpp"
#include "ActsPlugins/Root/TGeoLayerBuilder.hpp"
#include "ActsPlugins/Root/TGeoParser.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <map>
#include <numbers>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

#include "TGeoBBox.h"
#include "TGeoManager.h"
#include "TGeoMatrix.h"
#include "TGeoNode.h"
#include "TGeoVolume.h"

namespace o2::alice3
{

using namespace Acts;
// The Blueprint API (Blueprint, BlueprintNode, ContainerBlueprintNode,
// LayerBlueprintNode, GeometryIdentifierBlueprintNode) lives in
// Acts::Experimental in the ACTS version O2 builds against, and was later moved
// up into Acts, leaving deprecated aliases behind. Pulling in both namespaces
// keeps the unqualified names below resolving either way -- the alias and the
// class denote the same entity, so the lookup is not ambiguous.
using namespace Acts::Experimental;
using namespace Acts::UnitLiterals;
using namespace o2::alice3::gen3cfg;

namespace
{

// --------------------------------------------------------------------------
// build state
// --------------------------------------------------------------------------
/// Per-build state threaded through the helpers below.
///
/// NOTE (O2 port): the standalone actsO2 builder kept the config, the detector
/// elements and the passive surfaces in function-local statics, which made it
/// non-reentrant and tied their lifetime to the process. Inside a DPL device
/// that is not acceptable, so they live here and are owned by the caller.
struct BuildState {
  const Gen3GeometryConfig& cfg;
  std::vector<std::shared_ptr<Surface>>& passiveSurfaces;
};

// Generic knobs (not geometry-specific).
constexpr double kUnitScalor = 10.0; // TGeo cm -> ACTS mm

const ExtentEnvelope kLayerEnvelope =
  ExtentEnvelope{{.z = {2. * 1_mm, 2. * 1_mm}, .r = {2. * 1_mm, 2. * 1_mm}}};

// binning of the proto-material (material receiver) on the layer faces is
// geometry-config data: see cfg.matBinsPhi / matBinsZ / matBinsR.

// --------------------------------------------------------------------------
// surface loading + grouping
// --------------------------------------------------------------------------
// Non-const: setSurfaces() needs the non-const surface() overload (the const
// one yields shared_ptr<const Surface>, which will not convert).
using ElementPtr = std::shared_ptr<ActsPlugins::TGeoDetectorElement>;

/// One reconstruction layer: surfaces at a common radius (cylinder) or z (disc)
struct LayerGroup {
  bool isDisc = false;
  double key = 0.; // radius (cylinder) or z (disc), in mm
  std::vector<std::shared_ptr<Surface>> surfaces;
  std::string name; // set for passive layers; sensitive ones are auto-named
  bool isPassive = false;
  // Extra material z-extent from attached End-of-Stave cards;
  // sentinels (max < min) mean "none". Used by assignProtoMaterial to stretch
  // the representative cylinder in z.
  double matZMin = 1e9;
  double matZMax = -1e9;
};

/// Build the passive cylinder layers from the hardcoded table above.
std::vector<LayerGroup> makePassiveLayers(BuildState& st)
{
  std::vector<LayerGroup> out;
  for (const auto& pc : st.cfg.passiveCylinders) {
    auto surface = Surface::makeShared<CylinderSurface>(
      Transform3{Translation3{Vector3{0., 0., pc.zCentre * 1_mm}}},
      pc.r * 1_mm, pc.halfZ * 1_mm);
    st.passiveSurfaces.push_back(surface);
    LayerGroup g;
    g.isDisc = false;
    g.key = pc.r * 1_mm;
    g.surfaces = {surface};
    g.name = pc.name;
    g.isPassive = true;
    out.push_back(std::move(g));
  }
  return out;
}

/// Build the forward service shells for one side (negative or positive z).
std::vector<LayerGroup> makeForwardShells(BuildState& st, bool negative)
{
  std::vector<LayerGroup> out;
  for (const auto& fc : st.cfg.forwardCylinders) {
    const double halfZ = 0.5 * (fc.zMax - fc.zMin);
    const double zc = (negative ? -1.0 : 1.0) * 0.5 * (fc.zMax + fc.zMin);
    auto surface = Surface::makeShared<CylinderSurface>(
      Transform3{Translation3{Vector3{0., 0., zc * 1_mm}}}, fc.r * 1_mm,
      halfZ * 1_mm);
    st.passiveSurfaces.push_back(surface);
    LayerGroup g;
    g.isDisc = false;
    g.key = fc.r * 1_mm;
    g.surfaces = {surface};
    g.name = std::string{fc.name} + (negative ? "_neg" : "_pos");
    g.isPassive = true;
    out.push_back(std::move(g));
  }
  return out;
}

/// Build the passive service-disc layers from the hardcoded table above.
std::vector<LayerGroup> makePassiveDiscs(BuildState& st)
{
  std::vector<LayerGroup> out;
  for (const auto& pd : st.cfg.passiveDiscs) {
    auto surface = Surface::makeShared<DiscSurface>(
      Transform3{Translation3{Vector3{0., 0., pd.z * 1_mm}}}, pd.rMin * 1_mm,
      pd.rMax * 1_mm);
    st.passiveSurfaces.push_back(surface);
    LayerGroup g;
    g.isDisc = true;
    g.key = pd.z * 1_mm;
    g.surfaces = {surface};
    g.name = pd.name;
    g.isPassive = true;
    out.push_back(std::move(g));
  }
  return out;
}

/// Which detector a converted element belongs to.
///
/// Clustering is done per subsystem because no single tolerance works: merging
/// the staggered TRK rows (188.62/200.30, 11.68 mm apart) needs a tolerance
/// that would swallow ITOF (only 10.38 mm outside) and merge the vertex-detector
/// cylinders (7.00 mm apart). Each subsystem gets its own tolerance instead.
enum class Subsystem { VertexDetector,
                       TrkBarrel,
                       Itof,
                       Otof,
                       Ft3Disc };

/// Classify from the TGeo *volume* name (what TGeoParser matched against).
Subsystem classify(std::string_view volumeName)
{
  if (volumeName.find("PETALCASE") != std::string_view::npos) {
    return Subsystem::VertexDetector; // the 3 VD cylinders
  }
  if (volumeName.find("ITOFSensor") != std::string_view::npos) {
    return Subsystem::Itof;
  }
  if (volumeName.find("OTOFSensor") != std::string_view::npos) {
    return Subsystem::Otof;
  }
  if (volumeName.find("FT3Sensor") != std::string_view::npos) {
    return Subsystem::Ft3Disc;
  }
  return Subsystem::TrkBarrel; // TRKSensor0 chips
}

/// Radial (barrel) or longitudinal (disc) clustering tolerance per subsystem.
/// The values are geometry-specific and live in the config header (kTol*); the
/// mapping from Subsystem to value is behaviour and stays here.
double clusterTolerance(const BuildState& st, Subsystem s)
{
  switch (s) {
    case Subsystem::VertexDetector:
      return st.cfg.tolVertexDetector * 1_mm;
    case Subsystem::TrkBarrel:
      return st.cfg.tolTrkBarrel * 1_mm;
    case Subsystem::Itof:
      return st.cfg.tolItof * 1_mm;
    case Subsystem::Otof:
      return st.cfg.tolOtof * 1_mm;
    case Subsystem::Ft3Disc:
      return st.cfg.tolFt3Disc * 1_mm;
  }
  return st.cfg.tolVertexDetector * 1_mm;
}

/// One converted element together with the TGeo volume name it came from.
struct SensitiveElement {
  ElementPtr element;
  Subsystem subsystem;
};

/// Parse the TGeo file and convert all sensitive nodes into ACTS surfaces.
/// (No logging here: the ACTS_* macros need a local `logger()` accessor, which
/// is set up via ACTS_LOCAL_LOGGER in the entry point below.)
std::vector<SensitiveElement> loadSensitiveElements(TGeoManager& tgeo, const BuildState& st)
{
  std::vector<SensitiveElement> elements;

  // NOTE (O2 port): the standalone actsO2 builder called TGeoManager::Import()
  // here. Inside O2 the TGeo tree is the live gGeoManager shared with the rest
  // of the workflow, so it is passed in and must never be replaced.
  auto* topVolume = tgeo.GetTopVolume();
  if (topVolume == nullptr) {
    throw std::runtime_error("ALICE3 Gen3: no top volume in the TGeo geometry");
  }

  TGeoHMatrix gmatrix = TGeoIdentity(topVolume->GetName());

  ActsPlugins::TGeoParser::Options options;
  options.volumeNames = {topVolume->GetName()};
  options.targetNames = st.cfg.sensitiveMatches;
  options.unit = kUnitScalor;

  ActsPlugins::TGeoParser::State state;
  state.volume = topVolume;
  state.onBranch = true;

  ActsPlugins::TGeoParser::select(state, options, gmatrix);

  // Build the TGeoAxes (an ACTS type) from the config axis strings at runtime.
  const auto axesThinY = ActsPlugins::TGeoAxes::parse(st.cfg.axesThinY);
  const auto axesThinZ = ActsPlugins::TGeoAxes::parse(st.cfg.axesThinZ);

  elements.reserve(state.selectedNodes.size());
  for (const auto& snode : state.selectedNodes) {
    auto identifier = ActsPlugins::TGeoDetectorElement::Identifier();
    const Subsystem sub = classify(snode.node->GetVolume()->GetName());
    // barrel-type chips are thin in TGeo Y, the FT3 chips and the VD tubes
    // are handled by the default "XYZ" - see the axes* comment above
    const auto axes = (sub == Subsystem::TrkBarrel || sub == Subsystem::Itof ||
                       sub == Subsystem::Otof)
                        ? axesThinY
                        : axesThinZ;
    elements.push_back(SensitiveElement{
      ActsPlugins::TGeoLayerBuilder::defaultElementFactory(
        identifier, *snode.node, *snode.transform, axes, kUnitScalor,
        nullptr),
      sub});
  }

  if (elements.empty()) {
    throw std::runtime_error(
      "ALICE3 Gen3: no sensitive elements found in the TGeo geometry. Check "
      "'sensitiveMatches' in the Gen3 geometry config.");
  }
  return elements;
}

/// Group sorted (value, surface) pairs into clusters, splitting wherever two
/// CONSECUTIVE values differ by more than `tol`.
///
/// Gap-based (not "within tol of the first key"): order-independent, and a
/// cluster may be wider than `tol` - needed for OTOF, whose four staggered radii
/// span 6.36 mm but are only 1.5-2.8 mm apart consecutively.
std::vector<std::pair<double, std::vector<std::shared_ptr<Surface>>>> clusterBy(
  std::vector<std::pair<double, std::shared_ptr<Surface>>> items, double tol)
{
  std::vector<std::pair<double, std::vector<std::shared_ptr<Surface>>>> out;
  if (items.empty()) {
    return out;
  }
  std::ranges::sort(items, {}, &std::pair<double, std::shared_ptr<Surface>>::first);

  std::vector<std::shared_ptr<Surface>> current{items.front().second};
  double sum = items.front().first;
  double previous = items.front().first;

  auto flush = [&]() {
    out.emplace_back(sum / static_cast<double>(current.size()),
                     std::move(current));
    current.clear();
  };

  for (std::size_t i = 1; i < items.size(); ++i) {
    const double v = items[i].first;
    if (v - previous > tol) {
      flush();
      sum = 0.;
    }
    current.push_back(items[i].second);
    sum += v;
    previous = v;
  }
  flush();
  return out;
}

/// Cluster the surfaces into layers: cylinders by radius, discs by z, separately
/// within each subsystem (see Subsystem).
void groupSurfaces(const BuildState& st, const std::vector<SensitiveElement>& elements,
                   const GeometryContext& gctx,
                   std::vector<LayerGroup>& cylinders,
                   std::vector<LayerGroup>& discs)
{
  using Keyed = std::vector<std::pair<double, std::shared_ptr<Surface>>>;
  std::map<Subsystem, Keyed> cylByR, discByZ;

  for (const auto& [el, sub] : elements) {
    auto surf = el->surface().getSharedPtr();
    const auto stype = surf->type();

    if (stype == Surface::SurfaceType::Cylinder) {
      // vertex-detector layers: full cylinders (PETALCASE tubes)
      cylByR[sub].emplace_back(surf->bounds().values()[CylinderBounds::eR],
                               surf);
    } else if (stype == Surface::SurfaceType::Disc) {
      discByZ[sub].emplace_back(surf->center(gctx).z(), surf);
    } else {
      // planar sensors (TRK / ITOF / OTOF chips, FT3 petals):
      // normal mostly along z -> disc, else cylinder
      const auto c = surf->center(gctx);
      const auto n = surf->normal(gctx, c, Vector3::UnitZ());
      if (std::abs(n.z()) > 0.7) {
        discByZ[sub].emplace_back(c.z(), surf);
      } else {
        cylByR[sub].emplace_back(std::hypot(c.x(), c.y()), surf);
      }
    }
  }

  for (auto& [sub, keyed] : cylByR) {
    for (auto& [r, s] : clusterBy(std::move(keyed), clusterTolerance(st, sub))) {
      cylinders.push_back({false, r, std::move(s)});
    }
  }
  for (auto& [sub, keyed] : discByZ) {
    for (auto& [z, s] : clusterBy(std::move(keyed), clusterTolerance(st, sub))) {
      discs.push_back({true, z, std::move(s)});
    }
  }
  // subsystems were processed independently -> restore a global ordering
  std::ranges::sort(cylinders, {}, &LayerGroup::key);
  std::ranges::sort(discs, {}, &LayerGroup::key);
}

// --------------------------------------------------------------------------
// End-of-Stave passive readout cards
// --------------------------------------------------------------------------

/// Bounding-box extent of one End-of-Stave card, in ACTS mm (global frame).
struct EndOfStaveBox {
  double rMin, rMax, zMin, zMax;
};

/// Parse the passive End-of-Stave cards (kEndOfStaveMatches) and return the
/// bounding-box r/z extent of each in global mm. They are NOT converted to
/// surfaces - only their extent is needed to stretch a barrel layer's material
/// representative in z. Empty match list -> no-op. Reuses the TGeoManager
/// already imported by loadSensitiveElements.
std::vector<EndOfStaveBox> loadEndOfStaveExtents(TGeoManager& tgeo, const BuildState& st)
{
  std::vector<EndOfStaveBox> out;
  if (st.cfg.endOfStaveMatches.empty()) {
    return out;
  }
  auto* topVolume = tgeo.GetTopVolume();
  if (topVolume == nullptr) {
    return out;
  }
  TGeoHMatrix gmatrix = TGeoIdentity(topVolume->GetName());
  ActsPlugins::TGeoParser::Options options;
  options.volumeNames = {topVolume->GetName()};
  options.targetNames = st.cfg.endOfStaveMatches;
  options.unit = kUnitScalor;
  ActsPlugins::TGeoParser::State state;
  state.volume = topVolume;
  state.onBranch = true;
  ActsPlugins::TGeoParser::select(state, options, gmatrix);

  for (const auto& snode : state.selectedNodes) {
    // bounding box in the volume's local frame (cm), transformed to global
    auto* bb = dynamic_cast<TGeoBBox*>(snode.node->GetVolume()->GetShape());
    if (bb == nullptr) {
      continue; // non-box shapes (e.g. composites) skipped
    }
    const double dx = bb->GetDX(), dy = bb->GetDY(), dz = bb->GetDZ();
    const double* o = bb->GetOrigin();
    EndOfStaveBox e{1e9, -1e9, 1e9, -1e9};
    for (int sx = -1; sx <= 1; sx += 2) {
      for (int sy = -1; sy <= 1; sy += 2) {
        for (int sz = -1; sz <= 1; sz += 2) {
          const double loc[3] = {o[0] + sx * dx, o[1] + sy * dy, o[2] + sz * dz};
          double glo[3] = {0., 0., 0.};
          snode.transform->LocalToMaster(loc, glo);
          const double r = std::hypot(glo[0], glo[1]) * kUnitScalor;
          const double z = glo[2] * kUnitScalor;
          e.rMin = std::min(e.rMin, r);
          e.rMax = std::max(e.rMax, r);
          e.zMin = std::min(e.zMin, z);
          e.zMax = std::max(e.zMax, z);
        }
      }
    }
    out.push_back(e);
  }
  return out;
}

/// Attach each End-of-Stave card to the barrel SENSITIVE cylinder layer whose
/// radius lies within kEndOfStaveRTol of the card's radial span, recording the
/// card's z-extent as a material-extent hint on that layer.
void attachEndOfStave(const BuildState& st, std::vector<LayerGroup>& cylinders,
                      const std::vector<EndOfStaveBox>& boxes)
{
  for (const auto& b : boxes) {
    LayerGroup* best = nullptr;
    double bestD = st.cfg.endOfStaveRTol;
    for (auto& g : cylinders) {
      if (g.isPassive) {
        continue;
      }
      const double d = (g.key < b.rMin)   ? (b.rMin - g.key)
                       : (g.key > b.rMax) ? (g.key - b.rMax)
                                          : 0.0;
      if (d <= bestD) {
        bestD = d;
        best = &g;
      }
    }
    if (best != nullptr) {
      best->matZMin = std::min(best->matZMin, b.zMin);
      best->matZMax = std::max(best->matZMax, b.zMax);
    }
  }
}

// --------------------------------------------------------------------------
// blueprint assembly
// --------------------------------------------------------------------------

void configureContainer(ContainerBlueprintNode& node)
{
  node.setAttachmentStrategy(VolumeAttachmentStrategy::Gap);
  node.setResizeStrategies(VolumeResizeStrategy::Gap,
                           VolumeResizeStrategy::Gap);
}

/// R-stack whose children are NOT z-aligned (the forward regions).
///
/// `Gap` attachment must not be used: the inserted r-gap inherits its inner
/// neighbour's halfZ but is centred on the stack's FIRST child, so an off-centre
/// long neighbour makes the gap stick out and synchronizeZBounds() (a union over
/// all children, gaps included) inflates the region. For the forward regions the
/// gap between the two service shells dragged the inner edge from 1428 to 858 mm,
/// into Central -> "Volumes overlap in z". `Midpoint` expands the two neighbours
/// until they touch instead of inserting a volume, so no spurious child exists.
void configureUnalignedRContainer(ContainerBlueprintNode& node)
{
  node.setAttachmentStrategy(VolumeAttachmentStrategy::Midpoint);
  node.setResizeStrategies(VolumeResizeStrategy::Gap,
                           VolumeResizeStrategy::Gap);
}

/// Build the cylinder / disc surface that carries a planar-sensor layer's
/// material. Only for planar layers - for one already a cylinder/disc it would
/// land exactly on top of the existing surface. See assignProtoMaterial().
std::shared_ptr<Surface> makeRepresentativeSurface(bool isDisc, double rmin,
                                                   double rmax, double zmin,
                                                   double zmax)
{
  if (isDisc) {
    const double z = 0.5 * (zmin + zmax);
    return Surface::makeShared<DiscSurface>(
      Transform3{Translation3{Vector3{0., 0., z}}}, rmin, rmax);
  }
  const double r = 0.5 * (rmin + rmax);
  const double zc = 0.5 * (zmin + zmax);
  const double hz = 0.5 * (zmax - zmin);
  return Surface::makeShared<CylinderSurface>(
    Transform3{Translation3{Vector3{0., 0., zc}}}, r, hz);
}

/// Mark a layer's surfaces as binned material receivers (proto material).
///
/// Proto material is put on the surfaces directly, NOT via
/// MaterialDesignatorBlueprintNode::configureFace(): portal-face material is
/// rejected when the face is merged during stacking. It also must NOT go on the
/// sensitive planar sensors (ConvexPolygonBounds, which adjustBinUtility()
/// rejects), so a planar layer gets a synthetic cylinder/disc receiver instead
/// - see the needsRepresentative branch. These receivers, plus the passive
/// cylinders/discs, are what extractMaterialSurfaces() returns for mapping.
void assignProtoMaterial(BuildState& st, LayerGroup& group, const GeometryContext& gctx)
{
  constexpr float kPi = std::numbers::pi_v<float>;

  double rmin = 1e9, rmax = -1e9, zmin = 1e9, zmax = -1e9;
  for (const auto& s : group.surfaces) {
    const auto c = s->center(gctx);
    const auto vals = s->bounds().values();
    if (s->type() == Surface::SurfaceType::Cylinder && vals.size() >= 2) {
      const double r = vals[CylinderBounds::eR];
      const double hz = vals[CylinderBounds::eHalfLengthZ];
      rmin = std::min(rmin, r);
      rmax = std::max(rmax, r);
      zmin = std::min(zmin, c.z() - hz);
      zmax = std::max(zmax, c.z() + hz);
    } else if (s->type() == Surface::SurfaceType::Disc && vals.size() >= 2) {
      rmin = std::min(rmin, vals[0]);
      rmax = std::max(rmax, vals[1]);
      zmin = std::min(zmin, c.z());
      zmax = std::max(zmax, c.z());
    } else {
      // Planar sensor (stave / petal). Use the true corner positions, not the
      // centre: an FT3 petal spans ~200..680 mm in r, so a centre-only extent
      // would give a badly undersized representative disc.
      const auto poly = s->polyhedronRepresentation(gctx, 1u);
      for (const auto& v : poly.vertices) {
        const double r = std::hypot(v.x(), v.y());
        rmin = std::min(rmin, r);
        rmax = std::max(rmax, r);
        zmin = std::min(zmin, v.z());
        zmax = std::max(zmax, v.z());
      }
    }
  }
  // Approach: stretch the material extent along z to cover the End-of-Stave
  // cards attached to this barrel layer (their bounding box reaches past the
  // outermost sensor). Affects both the z-binning and the representative's
  // half-length below. No-op unless a card was attached (see attachEndOfStave).
  if (group.matZMax > group.matZMin) {
    zmin = std::min(zmin, group.matZMin);
    zmax = std::max(zmax, group.matZMax);
  }

  // guard against degenerate ranges (single-radius cylinder, single-z disc)
  if (!(rmax > rmin)) {
    rmin -= 1.;
    rmax += 1.;
  }
  if (!(zmax > zmin)) {
    zmin -= 1.;
    zmax += 1.;
  }

  BinUtility bu;
  if (group.isDisc) {
    bu = BinUtility(st.cfg.matBinsR, static_cast<float>(rmin),
                    static_cast<float>(rmax), open, AxisDirection::AxisR);
    bu += BinUtility(st.cfg.matBinsPhi, -kPi, kPi, closed, AxisDirection::AxisPhi);
  } else {
    bu = BinUtility(st.cfg.matBinsPhi, -kPi, kPi, closed, AxisDirection::AxisPhi);
    bu += BinUtility(st.cfg.matBinsZ, static_cast<float>(zmin),
                     static_cast<float>(zmax), open, AxisDirection::AxisZ);
  }

  auto material = std::make_shared<const ProtoSurfaceMaterial>(bu);

  // A synthetic receiver is only needed for PLANAR sensors (adjustBinUtility()
  // rejects their ConvexPolygonBounds); cylinders/discs carry material directly.
  // Not cosmetic: a barrel layer is a single cylinder (rmin==rmax), so a
  // representative would land exactly on the sensor - two coincident cylinders
  // the navigator can't disambiguate, and the layer would drop out of the
  // propagated material tracks.
  const bool needsRepresentative =
    !group.isPassive &&
    std::ranges::any_of(group.surfaces, [](const auto& s) {
      return s->type() != Surface::SurfaceType::Cylinder &&
             s->type() != Surface::SurfaceType::Disc;
    });

  if (!needsRepresentative) {
    // passive layers, and sensitive layers already made of cylinders / discs
    for (const auto& s : group.surfaces) {
      s->assignSurfaceMaterial(material);
    }
    return;
  }

  // Planar-sensor layer: the material goes on a synthetic cylinder / disc
  // spanning the layer, NOT on the sensors. See the note above.
  auto representative =
    makeRepresentativeSurface(group.isDisc, rmin, rmax, zmin, zmax);
  representative->assignSurfaceMaterial(material);
  st.passiveSurfaces.push_back(representative);
  group.surfaces.push_back(representative);
}

void addLayer(BlueprintNode& parent, const std::string& nameIn,
              const LayerGroup& group)
{
  const std::string name = group.name.empty() ? nameIn : group.name;
  parent.addLayer(name, [&group](LayerBlueprintNode& layer) {
    layer.setSurfaces(group.surfaces);
    layer.setLayerType(group.isDisc ? LayerBlueprintNode::LayerType::Disc
                                    : LayerBlueprintNode::LayerType::Cylinder);
    layer.setEnvelope(kLayerEnvelope);
    // Pin x,y to the beam line: the default centre-of-gravity placement leaves a
    // tiny x/y residual for a phi-segmented layer, which
    // CylinderVolumeStack::checkVolumeAlignment rejects ("not aligned:
    // translation in x or y"). Keep z on - it places each disc at its own z.
    layer.setUseCenterOfGravity(false, false, true);
  });
}

} // namespace

// ---------------------------------------------------------------------------
// entry point
// ---------------------------------------------------------------------------
// Details on the workflow (how much of the following is the "official" ACTS
// Gen3 workflow, and how much is ours):
//
// The official workflow is only two obligatory things: build a tree of
// BlueprintNodes, then call root.construct() (which runs Build/Connect/Finalize
// and does all portal wiring, gap insertion, navigation policies and geo-id
// assignment). The steps of this function map onto that as:
//
//   (numbered in the order they run below)
//   # step                                              status
//   - -------------------------------------------------  --------------------
//   1 parse TGeo -> surfaces (loadSensitiveElements)     official plugin;
//                                                        axes selection is ours
//   2 cluster surfaces -> layers (groupSurfaces)         ours
//   3 proto-material via synthetic surfaces + passives   ours (workaround)
//   4 partition layers into tree regions (pick/discsIn)  ours (topology)
//   5 Blueprint root + envelope                          official
//   6 assemble container/layer tree + pin IDs            official API + use
//   7 root.construct()                                   official
//   8 geo-id table dump                                  ours (diagnostic)
//
// (Step 3 runs before assembly so each receiver is grouped into its layer.)
//
// The Gen3 construction engine is used cleanly and completely. The distance
// from "official" is almost entirely that we lack a detector description: if the
// ALICE 3 geometry were available to ACTS as DD4hep or via the TGeo blueprint
// backend, steps 2, 4 and most of the step-6 tweaks would disappear (they only
// exist because we reconstruct the layer/region structure from a bare
// o2sim_geometry.root that carries none), and only the material approach
// (step 3) would remain as a real deviation.
// ---------------------------------------------------------------------------

Gen3BlueprintBuilder::Gen3BlueprintBuilder(Config config) : mConfig(std::move(config)) {}

o2::acts::TrackingGeometryOutput Gen3BlueprintBuilder::build(TGeoManager& tgeo,
                                                             const Acts::GeometryContext& gctxIn)
{
  using enum AxisDirection;

  // defines a local `logger()` accessor used by the ACTS_* macros
  ACTS_LOCAL_LOGGER(getDefaultLogger("ALICE3Gen3", mConfig.logLevel));
  ACTS_INFO("ALICE3 Gen3: building tracking geometry from TGeo geometry '" << tgeo.GetName() << "'");

  // Load the geometry configuration (JSON) - the single source of truth for all
  // geometry-specific parameters.
  if (mConfig.geometryConfigFile.empty()) {
    throw std::runtime_error(
      "ALICE3 Gen3: no geometry config file given. Point Config::geometryConfigFile at the "
      "gen3_geometry_config.json shipped with the geometry.");
  }
  ACTS_INFO("ALICE3 Gen3: loading geometry config from " << mConfig.geometryConfigFile);
  const Gen3GeometryConfig cfg = loadGen3GeometryConfig(mConfig.geometryConfigFile);

  o2::acts::TrackingGeometryOutput output;
  std::vector<std::shared_ptr<Surface>> passiveSurfaces;
  BuildState st{cfg, passiveSurfaces};

  const auto& gctx = gctxIn;
  const bool withMaterial = mConfig.withMaterial;

  // === STEP 1: parse TGeo -> surfaces (official plugin; axes selection ours) =
  auto elements = loadSensitiveElements(tgeo, st);
  ACTS_INFO("ALICE3 Gen3: converted " << elements.size() << " sensitive detector elements");

  // === STEP 2: cluster surfaces -> layers (ours) =========================
  std::vector<LayerGroup> cylinders, discs;
  groupSurfaces(st, elements, gctx, cylinders, discs);

  // === STEP 3: proto-material via synthetic surfaces + passives (ours) ====
  // Done here, before assembly, so each receiver is grouped with its layer.
  // add the passive material cylinders and re-sort by radius, so that the
  // region assignment below places them automatically
  if (withMaterial) {
    for (auto& pg : makePassiveLayers(st)) {
      cylinders.push_back(std::move(pg));
    }
    std::ranges::sort(cylinders, {}, &LayerGroup::key);
    for (auto& pg : makePassiveDiscs(st)) {
      discs.push_back(std::move(pg));
    }
    std::ranges::sort(discs, {}, &LayerGroup::key);
    ACTS_INFO("ALICE3 Gen3: added " << cfg.passiveCylinders.size() << " passive cylinders and "
                                    << cfg.passiveDiscs.size() << " passive discs");

    // Approach: stretch barrel sensor-layer material extents in z to cover the
    // passive End-of-Stave cards (endOfStaveMatches). No-op if the list is empty.
    const auto eosBoxes = loadEndOfStaveExtents(tgeo, st);
    attachEndOfStave(st, cylinders, eosBoxes);
    if (!eosBoxes.empty()) {
      ACTS_INFO("ALICE3 Gen3: attached " << eosBoxes.size()
                                         << " End-of-Stave card(s) to barrel layer material extents");
    }

    // Attach the material receivers. NOTE: non-const - for a sensitive layer
    // this appends the synthetic representative cylinder/disc to g.surfaces.
    for (auto& g : cylinders) {
      assignProtoMaterial(st, g, gctx);
    }
    for (auto& g : discs) {
      assignProtoMaterial(st, g, gctx);
    }
    ACTS_INFO("ALICE3 Gen3: proto material assigned to " << cylinders.size() + discs.size() << " layers");
  }

  ACTS_INFO("ALICE3 Gen3: " << cylinders.size() << " cylinder layers, " << discs.size() << " disc layers");
  for (const auto& g : cylinders) {
    ACTS_DEBUG("  cylinder r=" << g.key << " n=" << g.surfaces.size()
                               << (g.isPassive ? "  [passive] " + g.name : std::string{"  [sensitive]"}));
  }
  for (const auto& g : discs) {
    ACTS_DEBUG("  disc     z=" << g.key << " n=" << g.surfaces.size()
                               << (g.isPassive ? "  [passive] " + g.name : std::string{"  [sensitive]"}));
  }

  // Keep the detector elements alive: their surfaces hold only a raw back-pointer
  // to them. Ownership goes out with the output, so the caller decides the lifetime.
  output.elementStore.reserve(elements.size());
  for (const auto& se : elements) {
    output.elementStore.push_back(se.element);
  }

  // === STEP 4: partition layers into tree regions (ours; topology) =======
  auto pick = [](const std::vector<LayerGroup>& in, auto&& pred) {
    std::vector<LayerGroup> out;
    std::ranges::copy_if(in, std::back_inserter(out), pred);
    return out;
  };

  const auto innerBarrel = pick(cylinders, [&cfg](const LayerGroup& g) { return g.key < cfg.rInnerCoreMax; });
  const auto midBarrel = pick(cylinders, [&cfg](const LayerGroup& g) {
    return g.key >= cfg.rInnerCoreMax && g.key < cfg.rMainMax;
  });
  const auto otof = pick(cylinders, [&cfg](const LayerGroup& g) { return g.key >= cfg.rMainMax; });

  auto discsIn = [&](double lo, double hi, bool negative) {
    return pick(discs, [lo, hi, negative](const LayerGroup& g) {
      const double az = std::abs(g.key);
      return az >= lo && az < hi && ((g.key < 0) == negative);
    });
  };
  const auto ft3InnerNeg = discsIn(0., cfg.zCentralMax, true);
  const auto ft3InnerPos = discsIn(0., cfg.zCentralMax, false);
  const auto ft3OuterNeg = discsIn(cfg.zCentralMax, cfg.zMainMax, true);
  const auto ft3OuterPos = discsIn(cfg.zCentralMax, cfg.zMainMax, false);

  // === STEP 5: Blueprint root + envelope (official) ======================
  Blueprint::Config bpCfg;
  // The inner r envelope pulls the world's rMin to 0, else there is a hole
  // around the beam line (innermost volume starts at r=3.01) and tracks from the
  // origin abort with NavigatorError:3 (NoStartVolume). PadBlueprintNode clamps
  // std::max(0, rMin - rEnv[0]), so any value >= 3.01 gives rMin=0; 20 is used
  // for headroom against the innermost volume moving outward.
  constexpr double kWorldRInnerEnvelope = 20. * 1_mm; // >= innermost rMin
  bpCfg.envelope = ExtentEnvelope{{.z = {20. * 1_mm, 20. * 1_mm}, .r = {kWorldRInnerEnvelope, 20. * 1_mm}}};
  Blueprint root{bpCfg};

  // Ordering predicates that make the layer IDs deterministic (otherwise they
  // follow traversal order).
  const auto byMidRadius = [](const TrackingVolume& a, const TrackingVolume& b) {
    auto midR = [](const TrackingVolume& v) {
      const auto& cb = dynamic_cast<const CylinderVolumeBounds&>(v.volumeBounds());
      using enum CylinderVolumeBounds::BoundValues;
      return 0.5 * (cb.get(eMinR) + cb.get(eMaxR));
    };
    return midR(a) < midR(b);
  };
  const auto byZ = [&gctx](const TrackingVolume& a, const TrackingVolume& b) {
    return a.localToGlobalTransform(gctx).translation().z() <
           b.localToGlobalTransform(gctx).translation().z();
  };

  /// Wrap a region in a GeometryIdentifierBlueprintNode: pin its volume ID and
  /// number its layers from 1 in the given order.
  auto pinIds = [](BlueprintNode& parent, GeometryIdentifier::Value volumeId,
                   const auto& compare) -> GeometryIdentifierBlueprintNode& {
    auto& node = parent.withGeometryIdentifier();
    node.setAllVolumeIdsTo(volumeId).incrementLayerIds(1).sortBy(compare);
    return node;
  };

  // Name a layer "<region>_L<i>", i from 0 within the region. The region prefix
  // avoids the duplicate-volume-name throw. CAUTION: i is NOT the pinned layer
  // ID (incrementLayerIds counts passives and shells too) - the identifier table
  // printed after construction is the authority for (volume, layer).
  auto layerName = [](const std::string& region, std::size_t i) {
    return region + "_L" + std::to_string(i);
  };

  // === STEP 6: assemble container/layer tree + pin IDs (official API + use) =
  auto& core = root.addCylinderContainer("Core", AxisR);
  configureContainer(core);

  auto& main = core.addCylinderContainer("Main", AxisZ);
  configureContainer(main);

  // Each forward region is an R-stack: an inner z-stack of the discs, plus the
  // forward service shells as radial neighbours at the same z.
  auto addForwardRegion = [&](const std::string& name, const std::vector<LayerGroup>& regionDiscs,
                              bool negative, GeometryIdentifier::Value volumeId) {
    if (regionDiscs.empty()) {
      return;
    }
    // The whole forward region - discs AND service shells - shares one volume
    // ID. Sorting by z numbers the discs outward from the interaction point and
    // puts the two service shells last.
    auto& fwd = pinIds(main, volumeId, byZ).addCylinderContainer(name, AxisR);
    // disc stack and service shells are not z-aligned -> no Gap attachment
    configureUnalignedRContainer(fwd);

    const std::string discRegion = negative ? "OuterDisksNeg" : "OuterDisksPos";
    auto& discStack = fwd.addCylinderContainer(discRegion, AxisZ);
    configureContainer(discStack);

    // Inner-z end must Expand, not Gap. The disc stack defines the region's
    // inner z edge, so it is resized to a bound it already has; update()'s gap
    // guard (`newMinZ < oldMinZ`, no tolerance) then makes a ~0-halfZ gap from a
    // 1e-13 sliver, whose two coincident disc portals abort with "Have no portal
    // for NegativeDisc". Expand absorbs the sliver. The outer-z end keeps Gap (it
    // spans out to |z|=3502). Inner end is maxZ for z<0, minZ for z>0.
    discStack.setResizeStrategies(negative ? VolumeResizeStrategy::Gap : VolumeResizeStrategy::Expand,
                                  negative ? VolumeResizeStrategy::Expand : VolumeResizeStrategy::Gap);
    for (std::size_t i = 0; i < regionDiscs.size(); ++i) {
      addLayer(discStack, layerName(discRegion, i), regionDiscs[i]);
    }

    if (withMaterial) {
      for (auto& shell : makeForwardShells(st, negative)) {
        assignProtoMaterial(st, shell, gctx);
        addLayer(fwd, shell.name, shell);
      }
    }
  };

  addForwardRegion("FwdNeg", ft3OuterNeg, true, cfg.volFwdNeg);

  auto& central = main.addCylinderContainer("Central", AxisR);
  configureContainer(central);

  auto& innerCore = central.addCylinderContainer("InnerCore", AxisZ);
  configureContainer(innerCore);

  // Each inner-FT3 side gets its own z-container so it can carry a volume ID;
  // no new geometry (the discs are contiguous in z).
  if (!ft3InnerNeg.empty()) {
    auto& ft3NegNode =
      pinIds(innerCore, cfg.volFt3InnerNeg, byZ).addCylinderContainer("MiddleDisksNeg", AxisZ);
    configureContainer(ft3NegNode);
    for (std::size_t i = 0; i < ft3InnerNeg.size(); ++i) {
      addLayer(ft3NegNode, layerName("MiddleDisksNeg", i), ft3InnerNeg[i]);
    }
  }

  auto& innerBarrelNode =
    pinIds(innerCore, cfg.volInnerBarrel, byMidRadius).addCylinderContainer("InnerBarrel", AxisR);
  configureContainer(innerBarrelNode);
  for (std::size_t i = 0; i < innerBarrel.size(); ++i) {
    addLayer(innerBarrelNode, layerName("InnerBarrel", i), innerBarrel[i]);
  }

  if (!ft3InnerPos.empty()) {
    auto& ft3PosNode =
      pinIds(innerCore, cfg.volFt3InnerPos, byZ).addCylinderContainer("MiddleDisksPos", AxisZ);
    configureContainer(ft3PosNode);
    for (std::size_t i = 0; i < ft3InnerPos.size(); ++i) {
      addLayer(ft3PosNode, layerName("MiddleDisksPos", i), ft3InnerPos[i]);
    }
  }

  if (!midBarrel.empty()) {
    auto& midNode = pinIds(central, cfg.volOuterTrackerBarrel, byMidRadius)
                      .addCylinderContainer("OuterTrackerBarrel", AxisR);
    configureContainer(midNode);
    for (std::size_t i = 0; i < midBarrel.size(); ++i) {
      addLayer(midNode, layerName("OuterTrackerBarrel", i), midBarrel[i]);
    }
  }

  addForwardRegion("FwdPos", ft3OuterPos, false, cfg.volFwdPos);

  // OTOF likewise needs a container of its own to carry the pinned ID.
  if (!otof.empty()) {
    auto& otofNode = pinIds(core, cfg.volOtof, byMidRadius).addCylinderContainer("OTOF", AxisR);
    configureContainer(otofNode);
    for (std::size_t i = 0; i < otof.size(); ++i) {
      addLayer(otofNode, layerName("OTOF", i), otof[i]);
    }
  }

  if (!mConfig.graphvizFile.empty()) {
    std::ofstream fh(mConfig.graphvizFile);
    root.graphviz(fh);
    ACTS_INFO("ALICE3 Gen3: blueprint written to " << mConfig.graphvizFile);
  }

  // === STEP 7: construct - runs Build/Connect/Finalize (official) =========
  BlueprintOptions options;
  auto trackingGeometry = root.construct(options, gctx, logger());
  ACTS_INFO("ALICE3 Gen3: tracking geometry constructed");

  // The passive cylinders/discs and the synthetic material receivers need no
  // external holder: Acts::TrackingVolume owns its surfaces by shared_ptr, so
  // they stay alive with the geometry. Only the detector elements do, because
  // the Surface -> element link is a raw back-pointer.

  // === STEP 8: geometry identifier table (ours; diagnostic) ==============
  // These are the (volume, layer) keys the digitisation / seeding json must
  // use. Printed at INFO because getting them wrong fails silently: a
  // GeometryHierarchyMap entry that matches nothing simply never fires.
  ACTS_INFO("ALICE3 Gen3: geometry identifier table");
  ACTS_INFO("   volume  layer  nSurfaces  name");
  trackingGeometry->apply([&](const TrackingVolume& volume) {
    const auto gid = volume.geometryId();
    const std::size_t nSurfaces = volume.surfaces().size();
    if (nSurfaces == 0) {
      return; // container / gap volume, nothing to address
    }
    std::stringstream ss;
    ss << std::setw(9) << gid.volume() << std::setw(7) << gid.layer() << std::setw(11) << nSurfaces
       << "  " << volume.volumeName();
    ACTS_INFO(ss.str());
  });

  output.geometry = std::shared_ptr<Acts::TrackingGeometry>(std::move(trackingGeometry));
  return output;
}

} // namespace o2::alice3
