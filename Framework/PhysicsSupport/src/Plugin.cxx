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
//
#include "Framework/Plugins.h"
#include "Framework/ServiceHandle.h"
#include "Framework/ServiceSpec.h"
#include "Framework/CommonServices.h"
#include "Framework/O2DatabasePDGPlugin.h"
#include "TDatabasePDG.h"
#include "SimulationDataFormat/O2DatabasePDG.h"

using namespace o2::framework;

Double_t O2DatabasePDGImpl::Mass(int pdg, bool& success)
{
  // wrap our own Mass function to expose it in the service
  return o2::O2DatabasePDG::Mass(pdg, success, this);
}

struct PDGSupport : o2::framework::ServicePlugin {
  o2::framework::ServiceSpec* create() final
  {
    return new ServiceSpec{
      .name = "database-pdg",
      .init = [](ServiceRegistryRef, DeviceState&, fair::mq::ProgOptions&) -> ServiceHandle {
        auto* wrapper = new o2::framework::O2DatabasePDG();
        auto* ptr = new o2::framework::O2DatabasePDGImpl();
        o2::O2DatabasePDG::addALICEParticles(ptr);
        wrapper->setInstance(ptr);
        return ServiceHandle{TypeIdHelpers::uniqueId<O2DatabasePDG>(), wrapper, ServiceKind::Serial, "database-pdg"};
      },
      .configure = CommonServices::noConfiguration(),
      .exit = [](ServiceRegistryRef, void* service) { (*reinterpret_cast<O2DatabasePDG*>(service))->Delete(); },
      .kind = ServiceKind::Serial};
  }
};

struct PDGSupportOld : o2::framework::ServicePlugin {
  o2::framework::ServiceSpec* create() final
  {
    return new ServiceSpec{
      .name = "database-pdg-old",
      .init = [](ServiceRegistryRef ref, DeviceState&, fair::mq::ProgOptions&) -> ServiceHandle {
        return ServiceHandle{TypeIdHelpers::uniqueId<TDatabasePDG>(), ref.get<O2DatabasePDG>().get(), ServiceKind::Serial, "database-pdg-old"};
      },
      .configure = CommonServices::noConfiguration(),
      .kind = ServiceKind::Serial};
  }
};

DEFINE_DPL_PLUGINS_BEGIN
DEFINE_DPL_PLUGIN_INSTANCE(PDGSupportOld, CustomService);
DEFINE_DPL_PLUGIN_INSTANCE(PDGSupport, CustomService);
DEFINE_DPL_PLUGINS_END
