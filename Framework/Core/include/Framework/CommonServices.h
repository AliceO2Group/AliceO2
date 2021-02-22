// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#ifndef O2_FRAMEWORK_COMMONSERVICES_H_
#define O2_FRAMEWORK_COMMONSERVICES_H_

#include "Framework/ServiceSpec.h"
#include "Framework/TypeIdHelpers.h"

namespace o2::framework
{

struct ThreadPool {
  int poolSize;
};

/// A few ServiceSpecs for services we know about and that / are needed by
/// everyone.
struct CommonServices {
  /// An helper for services which do not need any / much special initialization or
  /// configuration.
  template <typename I, typename T>
  static ServiceInit simpleServiceInit()
  {
    return [](ServiceRegistry&, DeviceState&, fair::mq::ProgOptions& options) -> ServiceHandle {
      return ServiceHandle{TypeIdHelpers::uniqueId<I>(), new T, ServiceKind::Serial, typeid(T).name()};
    };
  }

  /// An helper to transform Singletons in to services, optionally configuring them
  template <typename I, typename T>
  static ServiceInit singletonServiceInit()
  {
    return [](ServiceRegistry&, DeviceState&, fair::mq::ProgOptions& options) -> ServiceHandle {
      return ServiceHandle{TypeIdHelpers::uniqueId<I>(), T::instance(), ServiceKind::Serial, typeid(T).name()};
    };
  }

  static ServiceConfigureCallback noConfiguration()
  {
    return [](InitContext&, void* service) -> void* { return service; };
  }

  static ServiceSpec driverClientSpec();
  static ServiceSpec monitoringSpec();
  static ServiceSpec infologgerContextSpec();
  static ServiceSpec infologgerSpec();
  static ServiceSpec configurationSpec();
  static ServiceSpec controlSpec();
  static ServiceSpec rootFileSpec();
  static ServiceSpec parallelSpec();
  static ServiceSpec rawDeviceSpec();
  static ServiceSpec callbacksSpec();
  static ServiceSpec timesliceIndex();
  static ServiceSpec dataRelayer();
  static ServiceSpec tracingSpec();
  static ServiceSpec threadPool(int numWorkers);
  static ServiceSpec dataProcessingStats();

  static std::vector<ServiceSpec> defaultServices(int numWorkers = 0);
  static std::vector<ServiceSpec> requiredServices();
};

} // namespace o2::framework

#endif // O2_FRAMEWORK_COMMONSERVICES_H_
