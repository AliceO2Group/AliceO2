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
/// \file TrackingGeometryManager.h
/// \brief Process-wide provider of the ACTS tracking geometry
/// \author Paolo Butti
///

#ifndef ALICEO2_ACTS_TRACKINGGEOMETRYMANAGER_H
#define ALICEO2_ACTS_TRACKINGGEOMETRYMANAGER_H

#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>

#include "Acts/Utilities/Logger.hpp"

#include "ACTSInterface/ITrackingGeometryBuilder.h"

namespace o2::acts
{

/// Provider of the shared, immutable Acts::TrackingGeometry.
///
/// Follows the same pattern as o2::base::Propagator::Instance(),
/// o2::base::GRPGeomHelper::instance() and o2::its::GeometryTGeo::Instance():
/// a process-wide lazily-built singleton fed from the geometry O2 already has
/// live in memory. It fetches nothing by itself -- the caller is responsible for
/// making gGeoManager available first, normally by declaring
/// o2::base::GRPGeomRequest::Aligned and calling
/// o2::base::GRPGeomHelper::instance().checkUpdates(pc).
///
/// Typical use in a DPL task:
/// \code
///   void init(InitContext&) {
///     o2::base::GRPGeomHelper::instance().setRequest(mCCDBReq);
///     TrackingGeometryManager::instance().setBuilder(std::make_unique<MyBuilder>(cfg));
///   }
///   void run(ProcessingContext& pc) {
///     o2::base::GRPGeomHelper::instance().checkUpdates(pc);   // gGeoManager now live
///     const auto& tg = TrackingGeometryManager::instance().get();  // built once
///   }
/// \endcode
///
/// Thread safety: build() is not re-entrant and is expected to run once, before
/// any parallel processing. Once built the geometry is immutable and all
/// accessors are safe to call concurrently.
class TrackingGeometryManager
{
 public:
  static TrackingGeometryManager& instance();

  /// Install the detector-specific builder. Must be called before the first build.
  void setBuilder(std::unique_ptr<ITrackingGeometryBuilder> builder);
  bool hasBuilder() const { return mBuilder != nullptr; }

  /// Path to an ACTS JSON material map, applied after construction.
  /// Empty (the default) means no material decoration.
  void setMaterialMapFile(std::string fileName) { mMaterialMapFile = std::move(fileName); }
  const std::string& getMaterialMapFile() const { return mMaterialMapFile; }

  void setLogLevel(Acts::Logging::Level level) { mLogLevel = level; }

  /// Build from the live gGeoManager.
  /// \throw std::runtime_error if no builder is set or no geometry is loaded
  void build();

  /// Build from a TGeo file, for macros and tests running outside a workflow.
  /// Loads the file through o2::base::GeometryManager, so it ends up in
  /// gGeoManager exactly as it would inside a workflow.
  void buildFromFile(std::string_view geomFileName);

  bool isBuilt() const { return mGeometry != nullptr; }
  void clear();

  /// Builds on first call if not built yet.
  std::shared_ptr<const Acts::TrackingGeometry> getShared();
  const Acts::TrackingGeometry& get();

  /// Sensor <-> surface lookup, as filled by the builder. Empty if the builder
  /// did not provide one -- which is the normal case when the geometry spans
  /// several detectors, since each has its own sensor numbering. Use the
  /// overload below for those.
  const SurfaceIndexMap& getIndex();

  /// Sensor <-> surface lookup for one detector, built from its geometry helper
  /// on first request and cached per detector afterwards.
  ///
  /// Pass \p pathProvider whenever the detector offers one -- it makes the match an
  /// identity rather than a geometric inference, see makeSurfaceIndex():
  /// \code
  ///   auto* trkGeo = o2::trk::GeometryTGeo::Instance();
  ///   mgr.getIndex(*trkGeo, 1e-3,
  ///                [trkGeo](int id) { return std::string(trkGeo->getMatrixPath(id).Data()); });
  /// \endcode
  ///
  /// \param cache        a DetMatrixCache-derived helper with a filled L2G cache,
  ///                     e.g. o2::trk::GeometryTGeo::Instance()
  /// \param toleranceCm  matching tolerance, see makeSurfaceIndex()
  /// \param pathProvider optional TGeo path lookup for exact identification
  const SurfaceIndexMap& getIndex(const o2::detectors::DetMatrixCache& cache,
                                  double toleranceCm = 1e-3,
                                  SensorPathProvider pathProvider = {});

  /// Context the geometry was built with. Carries no alignment payload: O2 bakes
  /// alignment into the TGeo matrices before the ACTS geometry is built, so the
  /// nominal transforms are already the aligned ones.
  const Acts::GeometryContext& getNominalContext() const { return mNominalContext; }

  /// Number of surfaces that received material during decoration. Zero when no
  /// material map was configured.
  std::size_t getNDecoratedSurfaces() const { return mNDecoratedSurfaces; }

 private:
  TrackingGeometryManager();

  void doBuild();
  void applyMaterialMapImpl(Acts::TrackingGeometry& geometry);

  std::unique_ptr<ITrackingGeometryBuilder> mBuilder;
  std::string mMaterialMapFile;
  Acts::Logging::Level mLogLevel = Acts::Logging::INFO;

  Acts::GeometryContext mNominalContext;
  std::shared_ptr<const Acts::TrackingGeometry> mGeometry;
  std::vector<std::shared_ptr<const Acts::SurfacePlacementBase>> mElementStore;
  SurfaceIndexMap mIndex;
  /// Per-detector indices, keyed by o2::detectors::DetID.
  std::unordered_map<int, SurfaceIndexMap> mDetIndices;
  std::size_t mNDecoratedSurfaces = 0;
};

} // namespace o2::acts

#endif // ALICEO2_ACTS_TRACKINGGEOMETRYMANAGER_H
