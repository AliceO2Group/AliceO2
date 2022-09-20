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
#ifndef O2_FRAMEWORK_SERVICEREGISTRYREF_H_
#define O2_FRAMEWORK_SERVICEREGISTRYREF_H_

#include "Framework/ServiceRegistry.h"

namespace o2::framework
{
class ServiceRegistryRef
{
 public:
  ServiceRegistryRef(ServiceRegistry& registry)
    : mRegistry(registry)
  {
    auto tid = std::this_thread::get_id();
    std::hash<std::thread::id> hasher;
    mSalt = hasher(tid);
  }

  /// Check if service of type T is currently active.
  template <typename T>
  std::enable_if_t<std::is_const_v<T> == false, bool> active() const
  {
    constexpr auto typeHash = TypeIdHelpers::uniqueId<T>();
    if (mRegistry.getPos(typeHash, 0) != -1) {
      return true;
    }
    auto result = mRegistry.getPos(typeHash, mSalt) != -1;
    return result;
  }

  /// Get a service for the given interface T. The returned reference exposed to
  /// the user is actually of the last concrete type C registered, however this
  /// should not be a problem.
  template <typename T>
  T& get() const
  {
    constexpr auto typeHash = TypeIdHelpers::uniqueId<T>();
    auto ptr = mRegistry.get(typeHash, mSalt, ServiceKind::Serial, typeid(T).name());
    if (O2_BUILTIN_LIKELY(ptr != nullptr)) {
      if constexpr (std::is_const_v<T>) {
        return *reinterpret_cast<T const*>(ptr);
      } else {
        return *reinterpret_cast<T*>(ptr);
      }
    }
    mRegistry.throwError(runtime_error_f("Unable to find service of kind %s. Make sure you use const / non-const correctly.", typeid(T).name()));
    O2_BUILTIN_UNREACHABLE();
  }

 private:
  ServiceRegistry& mRegistry;
  uint64_t mSalt;
};

} // namespace o2::framework

#endif // O2_FRAMEWORK_SERVICEREGISTRY_H_
