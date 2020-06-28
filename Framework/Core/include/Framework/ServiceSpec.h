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

#include <functional>
#include <string>
#include <vector>

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

/// Handle to the service hash must be calculated
/// using TypeIdHelper::uniqueId<BaseClass>() so that
/// we can retrieve the service by its baseclass.
struct ServiceHandle {
  uint32_t hash;
  void* instance;
};

/// A callback to create a given Service.
using ServiceInit = std::function<ServiceHandle(ServiceRegistry&, DeviceState&, fair::mq::ProgOptions&)>;

/// A callback to configure a given Service. Notice that the
/// service itself is type erased and it's responsibility of
/// the configuration itself to cast it to the correct value
using ServiceConfigure = std::function<void*(InitContext&, void*)>;

/// The kind of service we are asking for
enum struct ServiceKind {
  /// A Service which is not thread safe, therefore all accesses to it must be mutexed.
  Serial,
  /// A Service which is thread safe and therefore can be used by many threads the same time without risk
  Global,
  /// A Service which is specific to a given thread in a thread pool
  Stream
};

/// A declaration of a service to be used by the associated DataProcessor
struct ServiceSpec {
  std::string name;
  ServiceInit init;
  ServiceConfigure configure;
  ServiceKind kind;
};

} // namespace o2::framework

#endif // O2_FRAMEWORK_SERVICESPEC_H_
