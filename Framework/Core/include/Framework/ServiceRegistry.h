// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#ifndef O2_FRAMEWORK_SERVICEREGISTRY_H_
#define O2_FRAMEWORK_SERVICEREGISTRY_H_

#include "Framework/CompilerBuiltins.h"
#include "Framework/TypeIdHelpers.h"

#include <algorithm>
#include <array>
#include <exception>
#include <functional>
#include <string>
#include <type_traits>
#include <typeinfo>

namespace o2::framework
{

/// Service registry to hold generic, singleton like, interfaces and retrieve
/// them by type.
class ServiceRegistry
{
  using ServicePtr = void*;
  using ConstServicePtr = void const*;
  /// The maximum distance a entry can be from the optimal slot.
  constexpr static int MAX_DISTANCE = 8;
  /// The number of slots in the hashmap.
  constexpr static int MAX_SERVICES = 256;
  /// The mask to use to calculate the initial slot id.
  constexpr static int MAX_SERVICES_MASK = MAX_SERVICES - 1;

 public:
  ServiceRegistry()
  {
    mServices.fill(std::make_pair(0L, nullptr));
    mConstServices.fill(std::make_pair(0L, nullptr));
  }
  // Register a service for the given interface T
  // with actual implementation C, i.e. C is derived from T.
  // Only one instance of type C can be registered per type T.
  // The fact we use a bare pointer indicates that the ownership
  // of the service still belongs to whatever created it, and is
  // not passed to the registry. It's therefore responsibility of
  // the creator of the service to properly dispose it.
  template <class I, class C>
  void registerService(C* service)
  {
    // This only works for concrete implementations of the type T.
    // We need type elision as we do not want to know all the services in
    // advance
    static_assert(std::is_base_of<I, C>::value == true,
                  "Registered service is not derived from declared interface");
    auto typeHash = TypeIdHelpers::uniqueId<std::decay_t<I>>();
    auto serviceId = typeHash & MAX_SERVICES_MASK;
    for (uint8_t i = 0; i < MAX_DISTANCE; ++i) {
      if (mServices[i + serviceId].second == nullptr) {
        mServices[i + serviceId].first = typeHash;
        mServices[i + serviceId].second = reinterpret_cast<ServicePtr>(service);
        return;
      }
    }
    O2_BUILTIN_UNREACHABLE();
  }

  template <class I, class C>
  void registerService(C const* service)
  {
    // This only works for concrete implementations of the type T.
    // We need type elision as we do not want to know all the services in
    // advance
    static_assert(std::is_base_of<I, C>::value == true,
                  "Registered service is not derived from declared interface");
    auto typeHash = TypeIdHelpers::uniqueId<std::decay_t<I>>();
    auto serviceId = typeHash & MAX_SERVICES_MASK;
    for (uint8_t i = 0; i < MAX_DISTANCE; ++i) {
      if (mConstServices[i + serviceId].second == nullptr) {
        mConstServices[i + serviceId].first = typeHash;
        mConstServices[i + serviceId].second = reinterpret_cast<ConstServicePtr>(service);
        return;
      }
    }
    O2_BUILTIN_UNREACHABLE();
  }

  /// Get a service for the given interface T. The returned reference exposed to
  /// the user is actually of the last concrete type C registered, however this
  /// should not be a problem.
  template <typename T>
  std::enable_if_t<std::is_const_v<T> == false, T&> get() const
  {
    auto typeHash = TypeIdHelpers::uniqueId<std::decay_t<T>>();
    auto serviceId = typeHash & MAX_SERVICES_MASK;
    for (uint8_t i = 0; i < MAX_DISTANCE; ++i) {
      if (mServices[i + serviceId].first == typeHash) {
        return *reinterpret_cast<T*>(mServices[i + serviceId].second);
      }
    }
    throw std::runtime_error(std::string("Unable to find service of kind ") +
                             typeid(T).name() +
                             " did you register one as non-const reference?");
  }
  /// Get a service for the given interface T. The returned reference exposed to
  /// the user is actually of the last concrete type C registered, however this
  /// should not be a problem.
  template <typename T>
  std::enable_if_t<std::is_const_v<T>, T&> get() const
  {
    auto typeHash = TypeIdHelpers::uniqueId<std::decay_t<T>>();
    auto serviceId = typeHash & MAX_SERVICES_MASK;
    for (uint8_t i = 0; i < MAX_DISTANCE; ++i) {
      if (mConstServices[i + serviceId].first == typeHash) {
        return *reinterpret_cast<T const*>(mConstServices[i + serviceId].second);
      }
    }
    throw std::runtime_error(std::string("Unable to find service of kind ") +
                             typeid(T).name() +
                             " did you register one as const reference?");
  }

 private:
  std::array<std::pair<uint32_t, ServicePtr>, MAX_SERVICES + MAX_DISTANCE> mServices;
  std::array<std::pair<uint32_t, ConstServicePtr>, MAX_SERVICES + MAX_DISTANCE> mConstServices;
};

} // namespace o2::framework

#endif // O2_FRAMEWORK_SERVICEREGISTRY_H_
