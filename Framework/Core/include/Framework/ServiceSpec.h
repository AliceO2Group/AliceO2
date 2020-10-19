// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#ifndef O2_FRAMEWORK_SERVICESPEC_H_
#define O2_FRAMEWORK_SERVICESPEC_H_

#include "Framework/ServiceHandle.h"
#include <functional>
#include <string>
#include <vector>

#include <boost/program_options/variables_map.hpp>

namespace fair::mq
{
struct ProgOptions;
}

namespace o2::framework
{

struct InitContext;
struct DeviceSpec;
struct ServiceRegistry;
struct DeviceState;
struct ProcessingContext;
struct EndOfStreamContext;
class DanglingContext;

/// A callback to create a given Service.
using ServiceInit = std::function<ServiceHandle(ServiceRegistry&, DeviceState&, fair::mq::ProgOptions&)>;

/// A callback to configure a given Service. Notice that the
/// service itself is type erased and it's responsibility of
/// the configuration itself to cast it to the correct value
using ServiceConfigureCallback = std::function<void*(InitContext&, void*)>;

/// A callback which is executed before each processing loop.
using ServiceProcessingCallback = std::function<void(ProcessingContext&, void*)>;

/// A callback which is executed before each dangling input loop
using ServiceDanglingCallback = std::function<void(DanglingContext&, void*)>;

/// A callback which is executed before the end of stream loop.
using ServiceEOSCallback = std::function<void(EndOfStreamContext&, void*)>;

/// Callback executed before the forking of a given device in the driver
/// Notice the forking can happen multiple times. It's responsibility of
/// the service to track how many times it happens and act accordingly.
using ServicePreFork = std::function<void(ServiceRegistry&, boost::program_options::variables_map const&)>;

/// Callback executed after forking a given device in the driver,
/// but before doing exec / starting the device.
using ServicePostForkChild = std::function<void(ServiceRegistry&)>;

/// Callback executed after forking a given device in the driver,
/// but before doing exec / starting the device.
using ServicePostForkParent = std::function<void(ServiceRegistry&)>;

/// Callback executed in the driver in order to process a metric.
using ServiceMetricHandling = std::function<void(ServiceRegistry&)>;

/// Callback executed in the child after dispatching happened.
using ServicePostDispatching = std::function<void(ProcessingContext&, void*)>;

/// A specification for a Service.
/// A Service is a utility class which does not perform
/// data processing itself, but it can be used by the data processor
/// to carry out common tasks (e.g. monitoring) or by the framework
/// to perform data processing related ancillary work (e.g. send
/// messages after a computation happended).
struct ServiceSpec {
  /// Name of the service
  std::string name;
  /// Callback to initialise the service.
  ServiceInit init;
  /// Callback to configure the service.
  ServiceConfigureCallback configure;
  /// Callback executed before actual processing happens.
  ServiceProcessingCallback preProcessing = nullptr;
  /// Callback executed once actual processing happened.
  ServiceProcessingCallback postProcessing = nullptr;
  /// Callback executed before the dangling inputs loop
  ServiceDanglingCallback preDangling = nullptr;
  /// Callback executed after the dangling inputs loop
  ServiceDanglingCallback postDangling = nullptr;
  /// Callback executed before the end of stream callback of the user happended
  ServiceEOSCallback preEOS = nullptr;
  /// Callback executed after the end of stream callback of the user happended
  ServiceEOSCallback postEOS = nullptr;
  /// Callback executed before the forking of a given device in the driver
  /// Notice the forking can happen multiple times. It's responsibility of
  /// the service to track how many times it happens and act accordingly.
  ServicePreFork preFork = nullptr;
  /// Callback executed after forking a given device in the driver,
  /// but before doing exec / starting the device.
  ServicePostForkChild postForkChild = nullptr;
  /// Callback executed after forking a given device in the driver,
  /// but before doing exec / starting the device.
  ServicePostForkParent postForkParent = nullptr;

  ///Callback executed after each metric is received by the driver.
  ServiceMetricHandling metricHandling = nullptr;

  /// Callback executed after a given input record has been successfully
  /// dispatched.
  ServicePostDispatching postDispatching = nullptr;

  /// Kind of service being specified.
  ServiceKind kind;
};

struct ServiceConfigureHandle {
  ServiceConfigureCallback callback;
  void* service;
};

struct ServiceProcessingHandle {
  ServiceProcessingCallback callback;
  void* service;
};

struct ServiceDanglingHandle {
  ServiceDanglingCallback callback;
  void* service;
};

struct ServiceEOSHandle {
  ServiceEOSCallback callback;
  void* service;
};

struct ServiceDispatchingHandle {
  ServicePostDispatching callback;
  void* service;
};

} // namespace o2::framework

#endif // O2_FRAMEWORK_SERVICESPEC_H_
