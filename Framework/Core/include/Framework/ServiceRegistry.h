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
#include "Framework/ServiceSpec.h"
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

struct MutexLock {
  void lock() { mutex.lock(); }
  void unlock() { mutex.unlock(); }
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

struct ServiceRegistry {
  /// The maximum distance a entry can be from the optimal slot.
  constexpr static int MAX_DISTANCE = 8;
  /// The number of slots in the hashmap.
  constexpr static int MAX_SERVICES = 256;
  /// The mask to use to calculate the initial slot id.
  constexpr static int MAX_SERVICES_MASK = MAX_SERVICES - 1;
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

 public:
  using hash_type = decltype(TypeIdHelpers::uniqueId<void>());
  ServiceRegistry();

  ServiceRegistry(ServiceRegistry const& other)
  {
    mServicesKey = other.mServicesKey;
    mServicesValue = other.mServicesValue;
    mServicesMeta = other.mServicesMeta;
    for (size_t i = 0; i < other.mServicesBooked.size(); ++i) {
      this->mServicesBooked[i] = other.mServicesBooked[i].load();
    }
  }

  ServiceRegistry& operator=(ServiceRegistry const& other)
  {
    mServicesKey = other.mServicesKey;
    mServicesValue = other.mServicesValue;
    mServicesMeta = other.mServicesMeta;
    for (size_t i = 0; i < other.mServicesBooked.size(); ++i) {
      this->mServicesBooked[i] = other.mServicesBooked[i].load();
    }
    return *this;
  }

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
  /// Declare a service by its ServiceSpec. If of type Global
  /// / Serial it will be immediately registered for tid 0,
  /// so that subsequent gets will ultimately use it.
  /// If it is of kind "Stream" we will create the Service only
  /// when requested by a given thread. This function is not
  /// thread safe.
  void declareService(ServiceSpec const& spec, DeviceState& state, fair::mq::ProgOptions& options);

  /// Bind the callbacks of a service spec to a given service.
  void bindService(ServiceSpec const& spec, void* service);

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

  /// Register a service given an handle
  void registerService(ServiceHandle handle)
  {
    auto tid = std::this_thread::get_id();
    std::hash<std::thread::id> hasher;
    ServiceRegistry::registerService(handle.hash, handle.instance, handle.kind, hasher(tid), handle.name.c_str());
  }

  mutable std::vector<ServiceSpec> mSpecs;
  mutable std::array<uint32_t, MAX_SERVICES + MAX_DISTANCE> mServicesKey;
  mutable std::array<void*, MAX_SERVICES + MAX_DISTANCE> mServicesValue;
  mutable std::array<ServiceMeta, MAX_SERVICES + MAX_DISTANCE> mServicesMeta;
  mutable std::array<std::atomic<bool>, MAX_SERVICES + MAX_DISTANCE> mServicesBooked;

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
    ServiceRegistry::registerService(typeHash, reinterpret_cast<void*>(service), K, hasher(tid), typeid(C).name());
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
    this->registerService(typeHash, reinterpret_cast<void*>(const_cast<C*>(service)), K, hasher(tid), typeid(C).name());
  }

  /// Check if service of type T is currently active.
  template <typename T>
  std::enable_if_t<std::is_const_v<T> == false, bool> active() const
  {
    constexpr auto typeHash = TypeIdHelpers::uniqueId<T>();
    auto tid = std::this_thread::get_id();
    std::hash<std::thread::id> hasher;
    if (this->getPos(typeHash, 0) != -1) {
      return true;
    }
    auto result = this->getPos(typeHash, hasher(tid)) != -1;
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
    auto ptr = this->get(typeHash, hasher(tid), ServiceKind::Serial, typeid(T).name());
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
};

} // namespace o2::framework

#endif // O2_FRAMEWORK_SERVICEREGISTRY_H_
