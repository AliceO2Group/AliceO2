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
#ifndef O2_FRAMEWORK_SERVICEREGISTRY_H_
#define O2_FRAMEWORK_SERVICEREGISTRY_H_

#include "Framework/ThreadSafetyAnalysis.h"
#include "Framework/ServiceHandle.h"
#include "Framework/ServiceSpec.h"
#include "Framework/ServiceRegistryHelpers.h"
#include "Framework/CompilerBuiltins.h"
#include "Framework/TypeIdHelpers.h"

#include <algorithm>
#include <array>
#include <functional>
#include <string>
#include <type_traits>
#include <typeinfo>
#include <thread>
#include <atomic>
#include <mutex>

namespace o2::framework
{

struct DeviceState;

struct ServiceMeta {
  ServiceKind kind = ServiceKind::Serial;
  uint64_t threadId = 0;
};

struct NoLocking {
  void lock() {}
  void unlock() {}
};

struct CAPABILITY("mutex") MutexLock {
  void lock() ACQUIRE() { mutex.lock(); }
  void unlock() RELEASE() { mutex.unlock(); }
  std::mutex& mutex;
};

// A pointer to a service. Includes the locking policy
// for that service.
template <typename T, typename LOCKING = NoLocking>
class service_ptr : LOCKING
{
 public:
  service_ptr(T* ptr, LOCKING policy) : LOCKING(policy), mPtr{ptr} { this->lock(); }
  ~service_ptr() { this->unlock(); }
  service_ptr(service_ptr<T, LOCKING> const&) = delete;
  service_ptr& operator=(service_ptr<T, LOCKING> const&) = delete;
  T& operator*() { return *mPtr; }
  T* operator->() { return mPtr; }

 private:
  T* mPtr;
};

template <typename, typename = void>
struct ServiceKindExtractor {
  constexpr static ServiceKind kind = ServiceKind::Serial;
};

template <typename T>
struct ServiceKindExtractor<T, std::void_t<decltype(T::service_kind)>> : std::is_same<decltype(T::service_kind), enum ServiceKind> {
  constexpr static ServiceKind kind = T::service_kind;
};

template <typename T>
inline constexpr ServiceKind service_kind_v = ServiceKindExtractor<T>::kind;

struct ServiceRegistry {
  struct Salt {
    short streamId = 0;
    short dataProcessorId = 0;
  };

  static constexpr Salt GLOBAL_CONTEXT_SALT{0, 0};

  struct InstanceId {
    uint32_t id = 0;
  };

  struct Index {
    int32_t index = -1;
  };

  struct SpecIndex {
    int index = -1;
  };

  // Metadata about the service. This
  // might be interesting for debugging purposes.
  // however it's not used to uniquely identify
  // the service.
  struct Meta {
    ServiceKind kind = ServiceKind::Serial;
    char const* name = nullptr;
    // The index in the
    SpecIndex specIndex{};
  };

  // Unique identifier for a service.
  // While we use the salted hash to find the bucket
  // in the hashmap, the service can be uniquely identified
  // only by this 64 bit value.
  struct Key {
    ServiceTypeHash typeHash;
    Salt salt;
  };

  static constexpr int32_t valueFromSalt(Salt salt) { return ((int32_t)salt.streamId) << 16 | salt.dataProcessorId; }
  static constexpr uint64_t valueFromKey(Key key) { return ((uint64_t)key.typeHash.hash) << 32 | ((uint64_t)valueFromSalt(key.salt)); }

  /// The maximum distance a entry can be from the optimal slot.
  constexpr static int32_t MAX_DISTANCE = 8;
  /// The number of slots in the hashmap.
  constexpr static uint32_t MAX_SERVICES = 256;
  /// The mask to use to calculate the initial slot id.
  constexpr static uint32_t MAX_SERVICES_MASK = MAX_SERVICES - 1;

  /// A salt which is global to the whole device.
  /// This can be used to query services which are not
  /// bound to a specific stream or data processor, e.g.
  /// the services to send metrics to the driver or
  /// to send messages to the control.
  static Salt globalDeviceSalt()
  {
    return GLOBAL_CONTEXT_SALT;
  }

  /// A salt which is global to a given stream
  /// but which multiple dataprocessors can share.
  static Salt globalStreamSalt(short streamId)
  {
    return {streamId, 0};
  }

  /// A salt which is global to a specific data processor.
  /// This can be used to query properties which are
  /// not bonded to a specific stream, e.g. the
  /// name of the data processor, its inputs and outputs,
  /// it's algorithm.
  static Salt dataProcessorSalt(short dataProcessorId)
  {
    // FIXME: old behaviour for now
    // return {0, dataProcessorId};
    return GLOBAL_CONTEXT_SALT;
  }

  /// A salt which is specific to a given stream.
  /// This can be used to query properties which are of the stream
  /// itself, e.g. the currently processed time frame by a given stream.
  static Salt streamSalt(short streamId, short dataProcessorId)
  {
    // FIXME: old behaviour for now
    // return {streamId, dataProcessorId};
    return {streamId, dataProcessorId};
  }

  constexpr InstanceId instanceFromTypeSalt(ServiceTypeHash type, Salt salt) const
  {
    return InstanceId{type.hash ^ valueFromSalt(salt)};
  }

  constexpr Index indexFromInstance(InstanceId id) const
  {
    static_assert(MAX_SERVICES_MASK < 0x7FFFFFFF, "MAX_SERVICES_MASK must be smaller than 0x7FFFFFFF");
    return Index{static_cast<int32_t>(id.id & MAX_SERVICES_MASK)};
  }

  /// Callbacks to be executed after the main GUI has been drawn
  mutable std::vector<ServicePostRenderGUIHandle> mPostRenderGUIHandles;

  /// To hide exception throwing from QC
  void throwError(const char* name, int64_t hash, int64_t streamId, int64_t dataprocessorId) const;

 public:
  using hash_type = decltype(TypeIdHelpers::uniqueId<void>());
  ServiceRegistry();

  ServiceRegistry(ServiceRegistry const& other);
  ServiceRegistry& operator=(ServiceRegistry const& other);

  /// Invoke callbacks on exit.
  void preExitCallbacks();

  /// Invoke after rendering the GUI. Can be used to
  /// add custom GUI elements associated to a given service.
  void postRenderGUICallbacks();

  /// Declare a service by its ServiceSpec. If of type Global
  /// / Serial it will be immediately registered for tid 0,
  /// so that subsequent gets will ultimately use it.
  /// If it is of kind "Stream" we will create the Service only
  /// when requested by a given thread. This function is not
  /// thread safe.
  /// @a salt is used to create the service in the proper context
  /// FIXME: for now we create everything in the global context
  void declareService(ServiceSpec const& spec, DeviceState& state, fair::mq::ProgOptions& options, ServiceRegistry::Salt salt = ServiceRegistry::globalDeviceSalt());

  void bindService(ServiceRegistry::Salt salt, ServiceSpec const& spec, void* service) const;

  void lateBindStreamServices(DeviceState& state, fair::mq::ProgOptions& options, ServiceRegistry::Salt salt);

  /// Type erased service registration. @a typeHash is the
  /// hash used to identify the service, @a service is
  /// a type erased pointer to the service itself.
  /// This method is supposed to be thread safe
  void registerService(ServiceTypeHash typeHash, void* service, ServiceKind kind, Salt salt, char const* name = nullptr, ServiceRegistry::SpecIndex specIndex = SpecIndex{-1}) const;

  // Lookup a given @a typeHash for a given @a threadId at
  // a unique (per typeHash) location. There might
  // be other typeHash which sit in the same place, but
  // the if statement will rule them out. As long as
  // only one thread writes in a given i + id location
  // as guaranteed by the atomic, mServicesKey[i + id] will
  // either be 0 or the final value.
  // This method should NEVER register a new service, event when requested.
  int getPos(ServiceTypeHash typeHash, Salt salt) const;

  // Basic, untemplated API. This will require explicitly
  // providing the @a typeHash for the Service type,
  // there @a threadId of the thread asking the service
  // and the @a kind of service. This
  // method might trigger the registering of a new service
  // if the service is not a stream service and the global
  // zero service is available.
  // Use this API only if you know what you are doing.
  void* get(ServiceTypeHash typeHash, Salt salt, ServiceKind kind, char const* name = nullptr) const;

  /// Register a service given an handle
  void registerService(ServiceHandle handle, Salt salt = ServiceRegistry::globalDeviceSalt())
  {
    ServiceRegistry::registerService({handle.hash}, handle.instance, handle.kind, salt, handle.name.c_str());
  }

  mutable std::vector<ServiceSpec> mSpecs;
  mutable std::array<std::atomic<Key>, MAX_SERVICES + MAX_DISTANCE> mServicesKey;
  mutable std::array<void*, MAX_SERVICES + MAX_DISTANCE> mServicesValue;
  mutable std::array<Meta, MAX_SERVICES + MAX_DISTANCE> mServicesMeta;
  mutable std::array<std::atomic<bool>, MAX_SERVICES + MAX_DISTANCE> mServicesBooked;

  /// @deprecated old API to be substituted with the ServiceHandle one
  template <class I, class C, enum ServiceKind K = ServiceKind::Serial>
  void registerService(C* service, Salt salt = ServiceRegistry::globalDeviceSalt())
  {
    // This only works for concrete implementations of the type T.
    // We need type elision as we do not want to know all the services in
    // advance
    static_assert(std::is_base_of<I, C>::value == true,
                  "Registered service is not derived from declared interface");
    constexpr ServiceTypeHash typeHash{TypeIdHelpers::uniqueId<I>()};
    ServiceRegistry::registerService(typeHash, reinterpret_cast<void*>(service), K, salt, typeid(C).name());
  }

  /// @deprecated old API to be substituted with the ServiceHandle one
  template <class I, class C, enum ServiceKind K = ServiceKind::Serial>
  void registerService(C const* service, Salt salt = ServiceRegistry::globalDeviceSalt())
  {
    // This only works for concrete implementations of the type T.
    // We need type elision as we do not want to know all the services in
    // advance
    static_assert(std::is_base_of<I, C>::value == true,
                  "Registered service is not derived from declared interface");
    constexpr ServiceTypeHash typeHash{TypeIdHelpers::uniqueId<I const>()};
    this->registerService(typeHash, reinterpret_cast<void*>(const_cast<C*>(service)), K, salt, typeid(C).name());
  }

  /// Check if service of type T is currently active.
  template <typename T>
  std::enable_if_t<std::is_const_v<T> == false, bool> active(Salt salt) const
  {
    constexpr ServiceTypeHash typeHash{TypeIdHelpers::uniqueId<T>()};
    if (this->getPos(typeHash, GLOBAL_CONTEXT_SALT) != -1) {
      return true;
    }
    auto result = this->getPos(typeHash, salt) != -1;
    return result;
  }

  /// Get a service for the given interface T. The returned reference exposed to
  /// the user is actually of the last concrete type C registered, however this
  /// should not be a problem.
  template <typename T>
  T& get(Salt salt) const
  {
    constexpr ServiceTypeHash typeHash{TypeIdHelpers::uniqueId<T>()};
    auto ptr = this->get(typeHash, salt, ServiceKindExtractor<T>::kind, typeid(T).name());
    if (O2_BUILTIN_LIKELY(ptr != nullptr)) {
      if constexpr (std::is_const_v<T>) {
        return *reinterpret_cast<T const*>(ptr);
      } else {
        return *reinterpret_cast<T*>(ptr);
      }
    }
    throwError(typeid(T).name(), typeHash.hash, salt.streamId, salt.dataProcessorId);
    O2_BUILTIN_UNREACHABLE();
  }

  /// Callback invoked by the driver after rendering.
  /// FIXME: Needs to stay here for the moment as it is used by
  /// the driver and not by one of the data processors.
  void postRenderGUICallbacks(ServiceRegistryRef ref);
};

} // namespace o2::framework

#endif // O2_FRAMEWORK_SERVICEREGISTRY_H_
