// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#ifndef O2_FRAMEWORK_SERVICEREGISTRYHELPERS_H_
#define O2_FRAMEWORK_SERVICEREGISTRYHELPERS_H_

#include "Framework/ServiceHandle.h"
#include "Framework/TypeIdHelpers.h"

#include <algorithm>
#include <array>
#include <functional>
#include <string>
#include <type_traits>
#include <typeinfo>
#include <thread>

namespace o2::framework
{

/// Helpers for ServiceRegistry manipulations
struct ServiceRegistryHelpers {
 public:
  // Create an handle for a service for the given interface T
  // with actual implementation C, i.e. C is derived from T.
  // Only one instance of type C can be registered per type T.
  // The fact we use a bare pointer indicates that the ownership
  // of the service still belongs to whatever created it, and is
  // not passed to the registry. It's therefore responsibility of
  // the creator of the service to properly dispose it.
  template <class I, class C, enum ServiceKind K = ServiceKind::Serial>
  static auto handleForService(C* service) -> ServiceHandle
  {
    // This only works for concrete implementations of the type T.
    // We need type elision as we do not want to know all the services in
    // advance
    static_assert(std::is_const_v<I> == false,
                  "Service interface must not be const if service object is not const");
    static_assert(std::is_base_of<I, C>::value == true,
                  "Registered service is not derived from declared interface");
    constexpr auto typeHash = TypeIdHelpers::uniqueId<I>();
    return ServiceHandle{typeHash, reinterpret_cast<void*>(service), K, typeid(C).name()};
  }

  /// Same as above, but for const instances
  template <class I, class C, enum ServiceKind K = ServiceKind::Serial>
  static auto handleForService(C const* service) -> ServiceHandle
  {
    // This only works for concrete implementations of the type T.
    // We need type elision as we do not want to know all the services in
    // advance
    static_assert(std::is_const_v<I> == true,
                  "Service interface must be const if service object is const");
    static_assert(std::is_base_of<I, C>::value == true,
                  "Registered service is not derived from declared interface");
    constexpr auto typeHash = TypeIdHelpers::uniqueId<I>();
    return ServiceHandle{typeHash, reinterpret_cast<void*>(const_cast<C*>(service)), K, typeid(C).name()};
  }
};

} // namespace o2::framework

#endif // O2_FRAMEWORK_SERVICEREGISTRYHELPERS_H_
