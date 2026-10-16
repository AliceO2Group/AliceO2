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
/// \file ActsGeometryService.h
/// \brief DPL service exposing the ACTS tracking geometry through the ServiceRegistry
/// \author Paolo Butti
///

#ifndef ALICEO2_ACTS_ACTSGEOMETRYSERVICE_H
#define ALICEO2_ACTS_ACTSGEOMETRYSERVICE_H

#include <memory>
#include <vector>

#include "Acts/Geometry/TrackingGeometry.hpp"

#include "ACTSInterface/SurfaceIndexMap.h"
#include "ACTSInterface/TrackingGeometryManager.h"
#include "Framework/ServiceSpec.h"

namespace o2::acts
{

/// Registry-facing handle on the ACTS tracking geometry.
///
/// This is a thin shim over TrackingGeometryManager: both access paths hand out
/// the very same geometry object, so a workflow can mix them freely. What the
/// service adds is DPL-owned lifetime (the geometry is released on device exit
/// rather than at static destruction) and retrieval through the registry:
///
/// \code
///   auto& actsGeo = pc.services().get<o2::acts::ActsGeometryService>();
///   const auto& tg = actsGeo.getGeometry();
/// \endcode
///
/// Registered with ServiceKind::DeviceGlobal: one instance per device process,
/// shared by every data processor in it, and safe to read from several threads
/// once built.
class ActsGeometryService
{
 public:
  /// Builds on first use. Requires gGeoManager to be live and a builder to have
  /// been installed on TrackingGeometryManager.
  const Acts::TrackingGeometry& getGeometry() const;
  std::shared_ptr<const Acts::TrackingGeometry> getGeometryShared() const;

  const SurfaceIndexMap& getIndex() const;

  const Acts::GeometryContext& getNominalContext() const;

  bool isBuilt() const;

  /// Escape hatch for configuration (builder, material map) from init().
  static TrackingGeometryManager& getManager() { return TrackingGeometryManager::instance(); }
};

/// ServiceSpec for ActsGeometryService, to be put into
/// DataProcessorSpec::requiredServices.
o2::framework::ServiceSpec actsGeometryServiceSpec();

/// CommonServices::defaultServices() plus the ACTS geometry service. Use as the
/// requiredServices of a DataProcessorSpec that wants registry access.
std::vector<o2::framework::ServiceSpec> defaultServicesWithActsGeometry();

} // namespace o2::acts

#endif // ALICEO2_ACTS_ACTSGEOMETRYSERVICE_H
