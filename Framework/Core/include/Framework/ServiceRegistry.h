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
#include <stdexcept>

namespace o2::framework
{

/// Service registry to hold generic, singleton like, interfaces and retrieve
/// them by type.
class ServiceRegistry
{
  /// The maximum distance a entry can be from the optimal slot.
  constexpr static int MAX_DISTANCE = 8;
  /// The number of slots in the hashmap.
  constexpr static int MAX_SERVICES = 256;
  /// The mask to use to calculate the initial slot id.
  constexpr static int MAX_SERVICES_MASK = MAX_SERVICES - 1;

 public:
  using hash_type = decltype(TypeIdHelpers::uniqueId<void>());

  ServiceRegistry()
  {
    mServicesKey.fill(0L);
    mServicesValue.fill(nullptr);
  }

  /// Type erased service registration. @a typeHash is the
  /// hash used to identify the service, @a service is
  /// a type erased pointer to the service itself.
  void registerService(hash_type typeHash, void* service)
  {
    hash_type id = typeHash & MAX_SERVICES_MASK;
    for (uint8_t i = 0; i < MAX_DISTANCE; ++i) {
      if (mServicesValue[i + id] == nullptr) {
        mServicesKey[i + id] = typeHash;
        mServicesValue[i + id] = service;
        return;
      }
    }
    O2_BUILTIN_UNREACHABLE();
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
    constexpr hash_type typeHash = TypeIdHelpers::uniqueId<I>();
    registerService(typeHash, reinterpret_cast<void*>(service));
  }

  template <class I, class C>
  void registerService(C const* service)
  {
    // This only works for concrete implementations of the type T.
    // We need type elision as we do not want to know all the services in
    // advance
    static_assert(std::is_base_of<I, C>::value == true,
                  "Registered service is not derived from declared interface");
    constexpr auto typeHash = TypeIdHelpers::uniqueId<I const>();
    constexpr auto id = typeHash & MAX_SERVICES_MASK;
    for (uint8_t i = 0; i < MAX_DISTANCE; ++i) {
      if (mServicesValue[i + id] == nullptr) {
        mServicesKey[i + id] = typeHash;
        mServicesValue[i + id] = (void*)(service);
        return;
      }
    }
    O2_BUILTIN_UNREACHABLE();
  }

  /// Check if service of type T is currently active.
  template <typename T>
  std::enable_if_t<std::is_const_v<T> == false, bool> active() const
  {
    constexpr auto typeHash = TypeIdHelpers::uniqueId<T>();
    auto ptr = get(typeHash);
    return ptr != nullptr;
  }

  /// Get a service for the given interface T. The returned reference exposed to
  /// the user is actually of the last concrete type C registered, however this
  /// should not be a problem.
  template <typename T>
  T& get() const
  {
    constexpr auto typeHash = TypeIdHelpers::uniqueId<T>();
    auto ptr = get(typeHash);
    if (O2_BUILTIN_LIKELY(ptr != nullptr)) {
      if constexpr (std::is_const_v<T>) {
        return *reinterpret_cast<T const*>(ptr);
      } else {
        return *reinterpret_cast<T*>(ptr);
      }
    }
    throw std::runtime_error(std::string("Unable to find service of kind ") +
                             typeid(T).name() +
                             ". Make sure you use const / non-const correctly.");
  }

  void* get(uint32_t typeHash) const
  {
    auto id = typeHash & MAX_SERVICES_MASK;
    for (uint8_t i = 0; i < MAX_DISTANCE; ++i) {
      if (mServicesKey[i + id] == typeHash) {
        return mServicesValue[i + id];
      }
    }
    return nullptr;
  }

 private:
  std::array<uint32_t, MAX_SERVICES + MAX_DISTANCE> mServicesKey;
  std::array<void*, MAX_SERVICES + MAX_DISTANCE> mServicesValue;
};

} // namespace o2::framework

#endif // O2_FRAMEWORK_SERVICEREGISTRY_H_
