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
/// \file ITrackingGeometryBuilder.h
/// \brief Detector-agnostic interface for building an Acts::TrackingGeometry from TGeo
/// \author Paolo Butti
///

#ifndef ALICEO2_ACTS_ITRACKINGGEOMETRYBUILDER_H
#define ALICEO2_ACTS_ITRACKINGGEOMETRYBUILDER_H

#include <memory>
#include <vector>

#include "Acts/Geometry/GeometryContext.hpp"
#include "Acts/Geometry/TrackingGeometry.hpp"
#include "Acts/Surfaces/SurfacePlacementBase.hpp"

#include "ACTSInterface/SurfaceIndexMap.h"

class TGeoManager;

namespace o2::acts
{

/// Everything a builder produces, kept together because the parts have a
/// lifetime dependency on each other.
struct TrackingGeometryOutput {
  /// Non-const so the manager can still decorate it with material before
  /// handing it out; it is published as shared_ptr<const> afterwards.
  std::shared_ptr<Acts::TrackingGeometry> geometry;

  /// Detector elements backing the sensitive surfaces.
  ///
  /// Acts::TrackingGeometry does NOT own these: Acts::Surface keeps a raw,
  /// non-owning back-pointer to its placement (Surface::surfacePlacement()).
  /// Destroying this store while \a geometry is alive is a dangling-pointer bug,
  /// so it must be kept for at least as long as the geometry.
  std::vector<std::shared_ptr<const Acts::SurfacePlacementBase>> elementStore;

  /// Sensor <-> surface lookup. May be empty if the builder cannot provide one.
  SurfaceIndexMap index;
};

/// Builds an Acts::TrackingGeometry from a TGeo tree.
///
/// Implementations are detector specific and live with their detector (see
/// o2::alice3::Gen3BlueprintBuilder). They are injected into
/// TrackingGeometryManager at runtime, so this library never depends on any
/// concrete detector.
class ITrackingGeometryBuilder
{
 public:
  virtual ~ITrackingGeometryBuilder() = default;

  /// Build from an already-loaded TGeo tree.
  ///
  /// \param tgeo the geometry manager to read; implementations must not import
  ///             or otherwise replace it, since it is normally O2's live
  ///             gGeoManager shared with the rest of the workflow
  /// \param gctx nominal context used while placing surfaces
  virtual TrackingGeometryOutput build(TGeoManager& tgeo, const Acts::GeometryContext& gctx) = 0;

  /// Short name, used in log messages.
  virtual const char* getName() const = 0;
};

} // namespace o2::acts

#endif // ALICEO2_ACTS_ITRACKINGGEOMETRYBUILDER_H
