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

#include "Framework/ServiceHandle.h"
#include "Framework/ServiceRegistryHelpers.h"
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
#include <thread>
#include <atomic>

namespace o2::framework
{

struct ServiceMeta {
  ServiceKind kind = ServiceKind::Serial;
  uint64_t threadId = 0;
};

struct ServiceRegistryBase {
  /// The maximum distance a entry can be from the optimal slot.
  constexpr static int MAX_DISTANCE = 8;
  /// The number of slots in the hashmap.
  constexpr static int MAX_SERVICES = 256;
  /// The mask to use to calculate the initial slot id.
  constexpr static int MAX_SERVICES_MASK = MAX_SERVICES - 1;

 public:
  using hash_type = decltype(TypeIdHelpers::uniqueId<void>());
  ServiceRegistryBase();

  ServiceRegistryBase(ServiceRegistryBase const& other)
  {
    mServicesKey = other.mServicesKey;
    mServicesValue = other.mServicesValue;
    mServicesMeta = other.mServicesMeta;
    for (size_t i = 0; i < other.mServicesBooked.size(); ++i) {
      this->mServicesBooked[i] = other.mServicesBooked[i].load();
    }
  }

  ServiceRegistryBase& operator=(ServiceRegistryBase const& other)
  {
    mServicesKey = other.mServicesKey;
    mServicesValue = other.mServicesValue;
    mServicesMeta = other.mServicesMeta;
    for (size_t i = 0; i < other.mServicesBooked.size(); ++i) {
      this->mServicesBooked[i] = other.mServicesBooked[i].load();
    }
    return *this;
  }

  /// Type erased service registration. @a typeHash is the
  /// hash used to identify the service, @a service is
  /// a type erased pointer to the service itself.
  /// This method is supposed to be thread safe
  void registerService(hash_type typeHash, void* service, ServiceKind kind, uint64_t threadId, char const* name = nullptr) const;

  // Lookup a given @a typeHash for a given @a threadId at
  // a unique (per typeHash) location. There might
  // be other typeHash which sit in the same place, but
  // the if statement will rule them out. As long as
  // only one thread writes in a given i + id location
  // as guaranteed by the atomic, mServicesKey[i + id] will
  // either be 0 or the final value.
  // This method should NEVER register a new service, event when requested.
  int getPos(uint32_t typeHash, uint64_t threadId) const
  {
    auto id = typeHash & MAX_SERVICES_MASK;
    auto threadHashId = (typeHash ^ threadId) & MAX_SERVICES_MASK;
    std::atomic_thread_fence(std::memory_order_acquire);
    for (uint8_t i = 0; i < MAX_DISTANCE; ++i) {
      if (mServicesKey[i + threadHashId] == typeHash) {
        return i + threadHashId;
      }
    }
    return -1;
  }

  // Basic, untemplated API. This will require explicitly
  // providing the @a typeHash for the Service type,
  // there @a threadId of the thread asking the service
  // and the @a kind of service. This
  // method might trigger the registering of a new service
  // if the service is not a stream service and the global
  // zero service is available.
  // Use this API only if you know what you are doing.
  void* get(uint32_t typeHash, uint64_t threadId, ServiceKind kind, char const* name = nullptr) const
  {
    // Look for the service. If found, return it.
    // Notice how due to threading issues, we might
    // find it with getPos, but the value can still
    // be nullptr.
    auto pos = getPos(typeHash, threadId);
    if (pos != -1) {
      void* ptr = mServicesValue[pos];
      if (ptr) {
        return ptr;
      }
    }
    // We are looking up a service which is not of
    // stream kind and was not looked up by this thread
    // before.
    if (threadId != 0) {
      int pos = getPos(typeHash, 0);
      if (pos != -1 && kind != ServiceKind::Stream) {
        registerService(typeHash, mServicesValue[pos], kind, threadId, name);
      }
      return mServicesValue[pos];
    }
    // If we are here it means we never registered a
    // service for the 0 thread (i.e. the main thread).
    return nullptr;
  }

  mutable std::array<uint32_t, MAX_SERVICES + MAX_DISTANCE> mServicesKey;
  mutable std::array<void*, MAX_SERVICES + MAX_DISTANCE> mServicesValue;
  mutable std::array<ServiceMeta, MAX_SERVICES + MAX_DISTANCE> mServicesMeta;
  mutable std::array<std::atomic<bool>, MAX_SERVICES + MAX_DISTANCE> mServicesBooked;
};

/// Service registry to hold generic, singleton like, interfaces and retrieve
/// them by type.
class ServiceRegistry : ServiceRegistryBase
{
 public:
  /// @deprecated old API to be substituted with the ServiceHandle one
  template <class I, class C, enum ServiceKind K = ServiceKind::Serial>
  void registerService(C* service)
  {
    // This only works for concrete implementations of the type T.
    // We need type elision as we do not want to know all the services in
    // advance
    static_assert(std::is_base_of<I, C>::value == true,
                  "Registered service is not derived from declared interface");
    constexpr hash_type typeHash = TypeIdHelpers::uniqueId<I>();
    auto tid = std::this_thread::get_id();
    std::hash<std::thread::id> hasher;
    ServiceRegistryBase::registerService(typeHash, reinterpret_cast<void*>(service), K, hasher(tid), typeid(C).name());
  }

  /// @deprecated old API to be substituted with the ServiceHandle one
  template <class I, class C, enum ServiceKind K = ServiceKind::Serial>
  void registerService(C const* service)
  {
    // This only works for concrete implementations of the type T.
    // We need type elision as we do not want to know all the services in
    // advance
    static_assert(std::is_base_of<I, C>::value == true,
                  "Registered service is not derived from declared interface");
    constexpr auto typeHash = TypeIdHelpers::uniqueId<I const>();
    constexpr auto id = typeHash & MAX_SERVICES_MASK;
    auto tid = std::this_thread::get_id();
    std::hash<std::thread::id> hasher;
    ServiceRegistryBase::registerService(typeHash, reinterpret_cast<void*>(const_cast<C*>(service)), K, hasher(tid), typeid(C).name());
  }

  /// Check if service of type T is currently active.
  template <typename T>
  std::enable_if_t<std::is_const_v<T> == false, bool> active() const
  {
    constexpr auto typeHash = TypeIdHelpers::uniqueId<T>();
    auto tid = std::this_thread::get_id();
    std::hash<std::thread::id> hasher;
    auto result = ServiceRegistryBase::getPos(typeHash, hasher(tid)) != -1;
    return result;
  }

  /// Get a service for the given interface T. The returned reference exposed to
  /// the user is actually of the last concrete type C registered, however this
  /// should not be a problem.
  template <typename T>
  T& get() const
  {
    constexpr auto typeHash = TypeIdHelpers::uniqueId<T>();
    auto tid = std::this_thread::get_id();
    std::hash<std::thread::id> hasher;
    auto ptr = ServiceRegistryBase::get(typeHash, hasher(tid), ServiceKind::Serial, typeid(T).name());
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

  /// Register a service given an handle
  void registerService(ServiceHandle handle)
  {
    auto tid = std::this_thread::get_id();
    std::hash<std::thread::id> hasher;
    ServiceRegistryBase::registerService(handle.hash, handle.instance, handle.kind, hasher(tid), handle.name.c_str());
  }
};

} // namespace o2::framework

#endif // O2_FRAMEWORK_SERVICEREGISTRY_H_
