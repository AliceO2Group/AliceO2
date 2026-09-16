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
/// \file ActsGeometryService.cxx
/// \author Paolo Butti
///

#include "ACTSWorkflow/ActsGeometryService.h"

#include "Framework/CommonServices.h"
#include "Framework/Logger.h"
#include "Framework/ServiceHandle.h"
#include "Framework/ServiceRegistryRef.h"
#include "Framework/TypeIdHelpers.h"

using namespace o2::framework;

namespace o2::acts
{

const Acts::TrackingGeometry& ActsGeometryService::getGeometry() const
{
  return TrackingGeometryManager::instance().get();
}

std::shared_ptr<const Acts::TrackingGeometry> ActsGeometryService::getGeometryShared() const
{
  return TrackingGeometryManager::instance().getShared();
}

const SurfaceIndexMap& ActsGeometryService::getIndex() const
{
  return TrackingGeometryManager::instance().getIndex();
}

const Acts::GeometryContext& ActsGeometryService::getNominalContext() const
{
  return TrackingGeometryManager::instance().getNominalContext();
}

bool ActsGeometryService::isBuilt() const
{
  return TrackingGeometryManager::instance().isBuilt();
}

ServiceSpec actsGeometryServiceSpec()
{
  return ServiceSpec{
    .name = "acts-geometry",
    .init = [](ServiceRegistryRef, DeviceState&, fair::mq::ProgOptions&) -> ServiceHandle {
      // Nothing is built here: the TGeo geometry is not available yet at service
      // init time. The first getGeometry() call, from run(), triggers the build.
      return ServiceHandle{TypeIdHelpers::uniqueId<ActsGeometryService>(), new ActsGeometryService(),
                           ServiceKind::DeviceGlobal, "acts-geometry"};
    },
    .configure = CommonServices::noConfiguration(),
    .exit = [](ServiceRegistryRef, void* service) {
      TrackingGeometryManager::instance().clear();
      delete reinterpret_cast<ActsGeometryService*>(service);
    },
    .kind = ServiceKind::DeviceGlobal};
}

std::vector<ServiceSpec> defaultServicesWithActsGeometry()
{
  auto services = CommonServices::defaultServices();
  services.push_back(actsGeometryServiceSpec());
  return services;
}

} // namespace o2::acts
