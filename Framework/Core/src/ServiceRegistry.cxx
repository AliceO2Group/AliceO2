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
#include "Framework/ServiceRegistry.h"
#include "Framework/ServiceRegistryRef.h"
#include "Framework/RawDeviceService.h"
#include "Framework/Tracing.h"
#include "Framework/Logger.h"
#include <fairmq/Device.h>
#include <iostream>

namespace o2::framework
{

ServiceRegistry::ServiceRegistry(ServiceRegistry const& other)
{
  for (size_t i = 0; i < MAX_SERVICES; ++i) {
    mServicesKey[i].store(other.mServicesKey[i].load());
  }
  mServicesValue = other.mServicesValue;
  mServicesMeta = other.mServicesMeta;
  for (size_t i = 0; i < other.mServicesBooked.size(); ++i) {
    this->mServicesBooked[i] = other.mServicesBooked[i].load();
  }
}

ServiceRegistry& ServiceRegistry::operator=(ServiceRegistry const& other)
{
  for (size_t i = 0; i < MAX_SERVICES; ++i) {
    mServicesKey[i].store(other.mServicesKey[i].load());
  }
  mServicesValue = other.mServicesValue;
  mServicesMeta = other.mServicesMeta;
  for (size_t i = 0; i < other.mServicesBooked.size(); ++i) {
    this->mServicesBooked[i] = other.mServicesBooked[i].load();
  }
  return *this;
}

ServiceRegistry::ServiceRegistry()
{
  for (size_t i = 0; i < MAX_SERVICES; ++i) {
    mServicesKey[i].store({0L, 0L});
  }

  mServicesValue.fill(nullptr);
  for (size_t i = 0; i < mServicesBooked.size(); ++i) {
    mServicesBooked[i] = false;
  }
}

/// Type erased service registration. @a typeHash is the
/// hash used to identify the service, @a service is
/// a type erased pointer to the service itself.
/// This method is supposed to be thread safe
void ServiceRegistry::registerService(ServiceTypeHash typeHash, void* service, ServiceKind kind, Salt salt, const char* name, SpecIndex specIndex) const
{
  LOGP(detail, "Registering service {} with hash {} in salt {} of kind {}",
       (name ? name : "<unknown>"),
       typeHash.hash,
       valueFromSalt(salt), (int)kind);
  if (specIndex.index == -1 && kind == ServiceKind::Stream && service == nullptr) {
    throw runtime_error_f("Cannot register a stream service %s without a valid spec index", name ? name : "<unknown>");
  }
  InstanceId id = instanceFromTypeSalt(typeHash, salt);
  Index index = indexFromInstance(id);
  // If kind is not stream, there is only one copy of our service.
  // So we look if it is already registered and reused it if it is.
  // If not, we register it as thread id 0 and as the passed one.
  if (kind != ServiceKind::Stream && salt.streamId != 0) {
    auto dataProcessorSalt = Salt{.streamId = GLOBAL_CONTEXT_SALT.streamId, .dataProcessorId = salt.dataProcessorId};
    void* oldService = this->get(typeHash, dataProcessorSalt, kind);
    if (oldService == nullptr) {
      registerService(typeHash, service, kind, dataProcessorSalt);
    } else {
      service = oldService;
    }
  }
  for (uint8_t i = 0; i < MAX_DISTANCE; ++i) {
    // If the service slot was not taken, take it atomically
    bool expected = false;
    if (mServicesBooked[i + index.index].compare_exchange_strong(expected, true,
                                                                 std::memory_order_seq_cst)) {
      mServicesValue[i + index.index] = service;
      mServicesMeta[i + index.index] = Meta{kind, name ? strdup(name) : nullptr, specIndex};
      mServicesKey[i + index.index] = Key{.typeHash = typeHash, .salt = salt};
      std::atomic_thread_fence(std::memory_order_release);
      return;
    }
  }
  throw runtime_error_f("Unable to find a spot in the registry for service %d. Make sure you use const / non-const correctly.", typeHash.hash);
}

void ServiceRegistry::declareService(ServiceSpec const& spec, DeviceState& state, fair::mq::ProgOptions& options, ServiceRegistry::Salt salt)
{
  mSpecs.push_back(spec);
  // Services which are not stream must have a single instance created upfront.
  if (spec.kind != ServiceKind::Stream) {
    ServiceHandle handle = spec.init({*this}, state, options);
    this->registerService({handle.hash}, handle.instance, handle.kind, salt, handle.name.c_str());
    this->bindService(spec, handle.instance);
  } else if (spec.kind == ServiceKind::Stream) {
    // We register a nullptr in this case, because we really want to have the ptr to
    // the service spec only.
    if (!spec.uniqueId) {
      throw runtime_error_f("Service %s is a stream service, but does not have a uniqueId method.", spec.name.c_str());
    }
    if (salt.streamId != 0) {
      throw runtime_error_f("Declaring a stream service %s in a non-global context is not allowed.", spec.name.c_str());
    }
    this->registerService({spec.uniqueId()}, nullptr, spec.kind, salt, spec.name.c_str(), {static_cast<int>(mSpecs.size() - 1)});
  }
}

void ServiceRegistry::bindService(ServiceSpec const& spec, void* service) const
{
  static TracyLockableN(std::mutex, bindMutex, "bind mutex");
  std::scoped_lock<LockableBase(std::mutex)> lock(bindMutex);
  if (spec.preProcessing) {
    mPreProcessingHandles.push_back(ServiceProcessingHandle{spec, spec.preProcessing, service});
  }
  if (spec.postProcessing) {
    mPostProcessingHandles.push_back(ServiceProcessingHandle{spec, spec.postProcessing, service});
  }
  if (spec.preDangling) {
    mPreDanglingHandles.push_back(ServiceDanglingHandle{spec, spec.preDangling, service});
  }
  if (spec.postDangling) {
    mPostDanglingHandles.push_back(ServiceDanglingHandle{spec, spec.postDangling, service});
  }
  if (spec.preEOS) {
    mPreEOSHandles.push_back(ServiceEOSHandle{spec, spec.preEOS, service});
  }
  if (spec.postEOS) {
    mPostEOSHandles.push_back(ServiceEOSHandle{spec, spec.postEOS, service});
  }
  if (spec.postDispatching) {
    mPostDispatchingHandles.push_back(ServiceDispatchingHandle{spec, spec.postDispatching, service});
  }
  if (spec.postForwarding) {
    mPostForwardingHandles.push_back(ServiceForwardingHandle{spec, spec.postForwarding, service});
  }
  if (spec.start) {
    mPreStartHandles.push_back(ServiceStartHandle{spec, spec.start, service});
  }
  if (spec.stop) {
    mPostStopHandles.push_back(ServiceStopHandle{spec, spec.stop, service});
  }
  if (spec.exit) {
    mPreExitHandles.push_back(ServiceExitHandle{spec, spec.exit, service});
  }
  if (spec.domainInfoUpdated) {
    mDomainInfoHandles.push_back(ServiceDomainInfoHandle{spec, spec.domainInfoUpdated, service});
  }
  if (spec.preSendingMessages) {
    mPreSendingMessagesHandles.push_back(ServicePreSendingMessagesHandle{spec, spec.preSendingMessages, service});
  }
  if (spec.postRenderGUI) {
    mPostRenderGUIHandles.push_back(ServicePostRenderGUIHandle{spec, spec.postRenderGUI, service});
  }
}

/// Invoke callbacks to be executed before every process method invokation
void ServiceRegistry::preProcessingCallbacks(ProcessingContext& processContext)
{
  for (auto& handle : mPreProcessingHandles) {
    handle.callback(processContext, handle.service);
  }
}
/// Invoke callbacks to be executed after every process method invokation
void ServiceRegistry::postProcessingCallbacks(ProcessingContext& processContext)
{
  for (auto& handle : mPostProcessingHandles) {
    handle.callback(processContext, handle.service);
  }
}
/// Invoke callbacks to be executed before every dangling check
void ServiceRegistry::preDanglingCallbacks(DanglingContext& danglingContext)
{
  for (auto preDanglingHandle : mPreDanglingHandles) {
    preDanglingHandle.callback(danglingContext, preDanglingHandle.service);
  }
}

/// Invoke callbacks to be executed after every dangling check
void ServiceRegistry::postDanglingCallbacks(DanglingContext& danglingContext)
{
  for (auto postDanglingHandle : mPostDanglingHandles) {
    LOGP(debug, "Doing postDanglingCallback for service {}", postDanglingHandle.spec.name);
    postDanglingHandle.callback(danglingContext, postDanglingHandle.service);
  }
}

/// Invoke callbacks to be executed before every EOS user callback invokation
void ServiceRegistry::preEOSCallbacks(EndOfStreamContext& eosContext)
{
  for (auto& eosHandle : mPreEOSHandles) {
    eosHandle.callback(eosContext, eosHandle.service);
  }
}

/// Invoke callbacks to be executed after every EOS user callback invokation
void ServiceRegistry::postEOSCallbacks(EndOfStreamContext& eosContext)
{
  for (auto& eosHandle : mPostEOSHandles) {
    eosHandle.callback(eosContext, eosHandle.service);
  }
}

/// Invoke callbacks to be executed after every data Dispatching
void ServiceRegistry::postDispatchingCallbacks(ProcessingContext& processContext)
{
  for (auto& dispatchingHandle : mPostDispatchingHandles) {
    dispatchingHandle.callback(processContext, dispatchingHandle.service);
  }
}

/// Invoke callbacks to be executed after every data Dispatching
void ServiceRegistry::postForwardingCallbacks(ProcessingContext& processContext)
{
  for (auto& forwardingHandle : mPostForwardingHandles) {
    forwardingHandle.callback(processContext, forwardingHandle.service);
  }
}

/// Callbacks to be called in fair::mq::Device::PreRun()
void ServiceRegistry::preStartCallbacks()
{
  // FIXME: we need to call the callback only once for the global services
  /// I guess...
  for (auto startHandle = mPreStartHandles.begin(); startHandle != mPreStartHandles.end(); ++startHandle) {
    startHandle->callback(*this, startHandle->service);
  }
}

void ServiceRegistry::postStopCallbacks()
{
  // FIXME: we need to call the callback only once for the global services
  /// I guess...
  for (auto& stopHandle : mPostStopHandles) {
    stopHandle.callback(*this, stopHandle.service);
  }
}

/// Invoke callback to be executed on exit, in reverse order.
void ServiceRegistry::preExitCallbacks()
{
  // FIXME: we need to call the callback only once for the global services
  /// I guess...
  for (auto exitHandle = mPreExitHandles.rbegin(); exitHandle != mPreExitHandles.rend(); ++exitHandle) {
    exitHandle->callback(ServiceRegistryRef{*this}, exitHandle->service);
  }
}

void ServiceRegistry::domainInfoUpdatedCallback(ServiceRegistry& registry, size_t oldestPossibleTimeslice, ChannelIndex channelIndex)
{
  for (auto& handle : mDomainInfoHandles) {
    handle.callback(*this, oldestPossibleTimeslice, channelIndex);
  }
}

void ServiceRegistry::preSendingMessagesCallbacks(ServiceRegistry& registry, fair::mq::Parts& parts, ChannelIndex channelIndex)
{
  for (auto& handle : mPreSendingMessagesHandles) {
    handle.callback(*this, parts, channelIndex);
  }
}

void ServiceRegistry::postRenderGUICallbacks()
{
  for (auto& handle : mPostRenderGUIHandles) {
    handle.callback(*this);
  }
}

void ServiceRegistry::throwError(RuntimeErrorRef const& ref) const
{
  throw ref;
}

int ServiceRegistry::getPos(ServiceTypeHash typeHash, Salt salt) const
{
  InstanceId instanceId = instanceFromTypeSalt(typeHash, salt);
  Index index = indexFromInstance(instanceId);
  for (uint8_t i = 0; i < MAX_DISTANCE; ++i) {
    if (valueFromKey(mServicesKey[i + index.index].load()) == valueFromKey({typeHash.hash, salt})) {
      return i + index.index;
    }
  }
  return -1;
}

void* ServiceRegistry::get(ServiceTypeHash typeHash, Salt salt, ServiceKind kind, char const* name) const
{
  // Cannot find a stream service using a global salt.
  if (salt.streamId == GLOBAL_CONTEXT_SALT.streamId && kind == ServiceKind::Stream) {
    throwError(runtime_error_f("Cannot find %s service using a global salt.", name ? name : "a stream"));
  }
  // Look for the service. If found, return it.
  // Notice how due to threading issues, we might
  // find it with getPos, but the value can still
  // be nullptr.
  auto pos = getPos(typeHash, salt);
  // If we are here it means we never registered a
  // service for the 0 thread (i.e. the main thread).
  if (pos != -1 && mServicesMeta[pos].kind == ServiceKind::Stream && valueFromSalt(mServicesKey[pos].load().salt) != valueFromSalt(salt)) {
    throwError(runtime_error_f("Inconsistent registry for thread %d. Expected %d", salt.streamId, mServicesKey[pos].load().salt.streamId));
    O2_BUILTIN_UNREACHABLE();
  }

  if (pos != -1) {
    bool isStream = mServicesMeta[pos].kind == ServiceKind::DataProcessorStream || mServicesMeta[pos].kind == ServiceKind::DeviceStream;
    bool isDataProcessor = mServicesMeta[pos].kind == ServiceKind::DataProcessorStream || mServicesMeta[pos].kind == ServiceKind::DataProcessorGlobal || mServicesMeta[pos].kind == ServiceKind::DataProcessorSerial;

    if (isStream && salt.streamId <= 0) {
      throwError(runtime_error_f("A stream service cannot be retrieved from a non stream salt %d", salt.streamId));
      O2_BUILTIN_UNREACHABLE();
    }

    if (isDataProcessor && salt.dataProcessorId < 0) {
      throwError(runtime_error_f("A data processor service cannot be retrieved from a non dataprocessor context %d", salt.dataProcessorId));
      O2_BUILTIN_UNREACHABLE();
    }

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
  if (salt.streamId == 0) {
    for (int i = 0; i < MAX_SERVICES; ++i) {
      if (mServicesKey[i].load().typeHash.hash == typeHash.hash && valueFromSalt(mServicesKey[i].load().salt) != valueFromSalt(salt)) {
        throwError(runtime_error_f("Service %s found in registry at %d rather than where expected by getPos", name, i));
      }
      if (mServicesKey[i].load().typeHash.hash == typeHash.hash) {
        throwError(runtime_error_f("Found service %s with hash %d but with salt %d of service kind %d",
                                   name, typeHash, valueFromSalt(mServicesKey[i].load().salt), (int)mServicesMeta[i].kind));
      }
    }
    throwError(runtime_error_f("Unable to find requested service %s with hash %d using salt %d for service kind %d",
                               name ? name : "<unknown>", typeHash, valueFromSalt(salt), (int)kind));
  }

  // Let's lookit up in the global context for the data processor.
  pos = getPos(typeHash, {.streamId = 0, .dataProcessorId = salt.dataProcessorId});
  if (pos != -1 && kind != ServiceKind::Stream) {
    // We found a global service. Register it for this stream and return it.
    // This will prevent ending up here in the future.
    LOGP(detail, "Caching global service {} for stream {}", name ? name : "", salt.streamId);
    mServicesKey[pos].load();
    std::atomic_thread_fence(std::memory_order_acquire);
    registerService(typeHash, mServicesValue[pos], kind, salt, name);
  }
  if (pos != -1) {
    mServicesKey[pos].load();
    std::atomic_thread_fence(std::memory_order_acquire);
    if (mServicesValue[pos]) {
      return mServicesValue[pos];
    }
    LOGP(detail, "Global service {} for stream {} is nullptr", name ? name : "", salt.streamId);
  }
  if (kind == ServiceKind::Stream) {
    LOGP(detail, "Registering a stream service {} with hash {} and salt {}", name ? name : "", typeHash.hash, valueFromSalt(salt));
    auto pos = getPos(typeHash, {.streamId = GLOBAL_CONTEXT_SALT.streamId, .dataProcessorId = salt.dataProcessorId});
    if (pos == -1) {
      throwError(runtime_error_f("Stream service %s with hash %d using salt %d for service kind %d was not declared upfront.",
                                 name, typeHash, valueFromSalt(salt), (int)kind));
    }
    auto& spec = mSpecs[mServicesMeta[pos].specIndex.index];
    auto& deviceState = this->get<DeviceState>(globalDeviceSalt());
    auto& rawDeviceService = this->get<RawDeviceService>(globalDeviceSalt());
    auto& registry = const_cast<ServiceRegistry&>(*this);
    // Call init for the proper ServiceRegistryRef
    ServiceHandle handle = spec.init({registry, salt}, deviceState, *rawDeviceService.device()->fConfig);
    this->registerService({handle.hash}, handle.instance, handle.kind, salt, handle.name.c_str());
    this->bindService(spec, handle.instance);
    return handle.instance;
  }

  LOGP(error, "Unable to find requested service {} with hash {} using salt {} for service kind {}", name ? name : "", typeHash.hash, valueFromSalt(salt), (int)kind);
  return nullptr;
}

} // namespace o2::framework
