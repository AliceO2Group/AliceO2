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
#include "Framework/RuntimeError.h"

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
  struct Context {
    short streamId = 0;
    short dataProcessorId = 0;
  };

  union Salt {
    Context context;
    int32_t value = 0;
  };

  static constexpr Salt GLOBAL_CONTEXT_SALT{0, 0};

  struct InstanceId {
    uint32_t id = 0;
  };

  struct Index {
    int32_t index = -1;
  };

  // Metadata about the service
  struct Meta {
    ServiceKind kind = ServiceKind::Serial;
    Salt salt = {0, 0};
  };

  /// The maximum distance a entry can be from the optimal slot.
  constexpr static int32_t MAX_DISTANCE = 8;
  /// The number of slots in the hashmap.
  constexpr static uint32_t MAX_SERVICES = 256;
  /// The mask to use to calculate the initial slot id.
  constexpr static uint32_t MAX_SERVICES_MASK = MAX_SERVICES - 1;

  static Salt threadSalt() {
    auto tid = std::this_thread::get_id();
    std::hash<std::thread::id> hasher;
    return Salt{Context{.streamId = (short)hasher(tid)}};
  }

  constexpr InstanceId instanceFromTypeSalt(ServiceTypeHash type, Salt salt) const
  {
    return InstanceId{type.hash ^ salt.value};
  }

  constexpr Index indexFromInstance(InstanceId id) const
  {
    static_assert(MAX_SERVICES_MASK < 0x7FFFFFFF, "MAX_SERVICES_MASK must be smaller than 0x7FFFFFFF");
    return Index{static_cast<int32_t>(id.id & MAX_SERVICES_MASK)};
  }

  /// Callbacks for services to be executed before every process method invokation
  std::vector<ServiceProcessingHandle> mPreProcessingHandles;
  /// Callbacks for services to be executed after every process method invokation
  std::vector<ServiceProcessingHandle> mPostProcessingHandles;
  /// Callbacks for services to be executed before every dangling check
  std::vector<ServiceDanglingHandle> mPreDanglingHandles;
  /// Callbacks for services to be executed after every dangling check
  std::vector<ServiceDanglingHandle> mPostDanglingHandles;
  /// Callbacks for services to be executed before every EOS user callback invokation
  std::vector<ServiceEOSHandle> mPreEOSHandles;
  /// Callbacks for services to be executed after every EOS user callback invokation
  std::vector<ServiceEOSHandle> mPostEOSHandles;
  /// Callbacks for services to be executed after every dispatching
  std::vector<ServiceDispatchingHandle> mPostDispatchingHandles;
  /// Callbacks for services to be executed after every dispatching
  std::vector<ServiceForwardingHandle> mPostForwardingHandles;
  /// Callbacks for services to be executed before Start
  std::vector<ServiceStartHandle> mPreStartHandles;
  /// Callbacks for services to be executed on the Stop transition
  std::vector<ServiceStopHandle> mPostStopHandles;
  /// Callbacks for services to be executed on exit
  std::vector<ServiceExitHandle> mPreExitHandles;
  /// Callbacks for services to be executed on exit
  std::vector<ServiceDomainInfoHandle> mDomainInfoHandles;
  /// Callbacks for services to be executed before sending messages
  std::vector<ServicePreSendingMessagesHandle> mPreSendingMessagesHandles;
  /// Callbacks to be executed after the main GUI has been drawn
  std::vector<ServicePostRenderGUIHandle> mPostRenderGUIHandles;

  /// To hide exception throwing from QC
  void throwError(RuntimeErrorRef const& ref) const;

 public:
  using hash_type = decltype(TypeIdHelpers::uniqueId<void>());
  ServiceRegistry();

  ServiceRegistry(ServiceRegistry const& other);
  ServiceRegistry& operator=(ServiceRegistry const& other);

  /// Invoke callbacks to be executed in PreRun(), before the User Start callbacks
  void preStartCallbacks();
  /// Invoke callbacks to be executed before every process method invokation
  void preProcessingCallbacks(ProcessingContext&);
  /// Invoke callbacks to be executed after every process method invokation
  void postProcessingCallbacks(ProcessingContext&);
  /// Invoke callbacks to be executed before every dangling check
  void preDanglingCallbacks(DanglingContext&);
  /// Invoke callbacks to be executed after every dangling check
  void postDanglingCallbacks(DanglingContext&);
  /// Invoke callbacks to be executed before every EOS user callback invokation
  void preEOSCallbacks(EndOfStreamContext&);
  /// Invoke callbacks to be executed after every EOS user callback invokation
  void postEOSCallbacks(EndOfStreamContext&);
  /// Invoke callbacks to monitor inputs after dispatching, regardless of them
  /// being discarded, consumed or processed.
  void postDispatchingCallbacks(ProcessingContext&);
  /// Callback invoked after the late forwarding has been done
  void postForwardingCallbacks(ProcessingContext&);

  /// Invoke callbacks on stop.
  void postStopCallbacks();
  /// Invoke callbacks on exit.
  void preExitCallbacks();

  /// Invoke whenever we get a new DomainInfo message
  void domainInfoUpdatedCallback(ServiceRegistry& registry, size_t oldestPossibleTimeslice, ChannelIndex channelIndex);

  /// Invoke before sending messages @a parts on a channel @a channelindex
  void preSendingMessagesCallbacks(ServiceRegistry& registry, fair::mq::Parts& parts, ChannelIndex channelindex);

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
  void declareService(ServiceSpec const& spec, DeviceState& state, fair::mq::ProgOptions& options, ServiceRegistry::Salt salt = ServiceRegistry::threadSalt());

  /// Bind the callbacks of a service spec to a given service.
  void bindService(ServiceSpec const& spec, void* service);

  /// Type erased service registration. @a typeHash is the
  /// hash used to identify the service, @a service is
  /// a type erased pointer to the service itself.
  /// This method is supposed to be thread safe
  void registerService(ServiceTypeHash typeHash, void* service, ServiceKind kind, Salt salt, char const* name = nullptr) const;

  // Lookup a given @a typeHash for a given @a threadId at
  // a unique (per typeHash) location. There might
  // be other typeHash which sit in the same place, but
  // the if statement will rule them out. As long as
  // only one thread writes in a given i + id location
  // as guaranteed by the atomic, mServicesKey[i + id] will
  // either be 0 or the final value.
  // This method should NEVER register a new service, event when requested.
  int getPos(ServiceTypeHash typeHash, Salt salt) const
  {
    InstanceId instanceId = instanceFromTypeSalt(typeHash, salt);
    Index index = indexFromInstance(instanceId);
    for (uint8_t i = 0; i < MAX_DISTANCE; ++i) {
      if (mServicesKey[i + index.index].load() == typeHash.hash) {
        return i + index.index;
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
  void* get(ServiceTypeHash typeHash, Salt salt, ServiceKind kind, char const* name = nullptr) const
  {
    // Look for the service. If found, return it.
    // Notice how due to threading issues, we might
    // find it with getPos, but the value can still
    // be nullptr.
    auto pos = getPos(typeHash, salt);
    if (pos != -1 && mServicesMeta[pos].kind == ServiceKind::Stream && mServicesMeta[pos].salt.value != salt.value) {
      throwError(runtime_error_f("Inconsistent registry for thread %d. Expected %d", salt.context.streamId, mServicesMeta[pos].salt.context.streamId));
      O2_BUILTIN_UNREACHABLE();
    }

    if (pos != -1) {
      mServicesKey[pos].load();
      std::atomic_thread_fence(std::memory_order_acquire);
      void* ptr = mServicesValue[pos];
      if (ptr) {
        return ptr;
      }
    }
    // We are looking up a service which is not of
    // stream kind and was not looked up by this thread
    // before.
    if (salt.value != GLOBAL_CONTEXT_SALT.value) {
      int pos = getPos(typeHash, GLOBAL_CONTEXT_SALT);
      if (pos != -1 && kind != ServiceKind::Stream) {
        mServicesKey[pos].load();
        std::atomic_thread_fence(std::memory_order_acquire);
        registerService(typeHash, mServicesValue[pos], kind, salt, name);
      }
      if (pos != -1) {
        mServicesKey[pos].load();
        std::atomic_thread_fence(std::memory_order_acquire);
        return mServicesValue[pos];
      } else {
        throwError(runtime_error_f("Unable to find requested service %s", name));
      }
    }
    // If we are here it means we never registered a
    // service for the 0 thread (i.e. the main thread).
    return nullptr;
  }

  /// Register a service given an handle
  void registerService(ServiceHandle handle, Salt salt)
  {
    ServiceRegistry::registerService({handle.hash}, handle.instance, handle.kind, salt, handle.name.c_str());
  }

  mutable std::vector<ServiceSpec> mSpecs;
  mutable std::array<std::atomic<uint32_t>, MAX_SERVICES + MAX_DISTANCE> mServicesKey;
  mutable std::array<void*, MAX_SERVICES + MAX_DISTANCE> mServicesValue;
  mutable std::array<Meta, MAX_SERVICES + MAX_DISTANCE> mServicesMeta;
  mutable std::array<std::atomic<bool>, MAX_SERVICES + MAX_DISTANCE> mServicesBooked;

  /// @deprecated old API to be substituted with the ServiceHandle one
  template <class I, class C, enum ServiceKind K = ServiceKind::Serial>
  void registerService(C* service, Salt salt)
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
  void registerService(C const* service, Salt salt)
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
    auto ptr = this->get(typeHash, salt, ServiceKind::Serial, typeid(T).name());
    if (O2_BUILTIN_LIKELY(ptr != nullptr)) {
      if constexpr (std::is_const_v<T>) {
        return *reinterpret_cast<T const*>(ptr);
      } else {
        return *reinterpret_cast<T*>(ptr);
      }
    }
    throwError(runtime_error_f("Unable to find service of kind %s. Make sure you use const / non-const correctly.", typeid(T).name()));
    O2_BUILTIN_UNREACHABLE();
  }
};

} // namespace o2::framework

#endif // O2_FRAMEWORK_SERVICEREGISTRY_H_
