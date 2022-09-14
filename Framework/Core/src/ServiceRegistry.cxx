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
#include "Framework/Tracing.h"
#include "Framework/Logger.h"
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
    mServicesKey[i].store(0L);
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
void ServiceRegistry::registerService(hash_type typeHash, void* service, ServiceKind kind, uint64_t threadId, const char* name) const
{
  hash_type id = typeHash & MAX_SERVICES_MASK;
  hash_type threadHashId = (typeHash ^ threadId) & MAX_SERVICES_MASK;
  // If kind is not stream, there is only one copy of our service.
  // So we look if it is already registered and reused it if it is.
  // If not, we register it as thread id 0 and as the passed one.
  if (kind != ServiceKind::Stream && threadId != 0) {
    void* oldService = this->get(typeHash, 0, kind);
    if (oldService == nullptr) {
      registerService(typeHash, service, kind, 0);
    } else {
      service = oldService;
    }
  }
  for (uint8_t i = 0; i < MAX_DISTANCE; ++i) {
    // If the service slot was not taken, take it atomically
    bool expected = false;
    if (mServicesBooked[i + threadHashId].compare_exchange_strong(expected, true,
                                                                  std::memory_order_seq_cst)) {
      mServicesValue[i + threadHashId] = service;
      mServicesMeta[i + threadHashId] = ServiceMeta{kind, threadId};
      mServicesKey[i + threadHashId] = typeHash;
      std::atomic_thread_fence(std::memory_order_release);
      return;
    }
  }
  throw std::runtime_error(std::string("Unable to find a spot in the registry for service ") +
                           std::to_string(typeHash) +
                           ". Make sure you use const / non-const correctly.");
}

void ServiceRegistry::declareService(ServiceSpec const& spec, DeviceState& state, fair::mq::ProgOptions& options)
{
  mSpecs.push_back(spec);
  // Services which are not stream must have a single instance created upfront.
  if (spec.kind != ServiceKind::Stream) {
    ServiceHandle handle = spec.init(*this, state, options);
    this->registerService(handle.hash, handle.instance, handle.kind, 0, handle.name.c_str());
    this->bindService(spec, handle.instance);
  }
}

void ServiceRegistry::bindService(ServiceSpec const& spec, void* service)
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
    exitHandle->callback(*this, exitHandle->service);
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

void ServiceRegistry::throwError(RuntimeErrorRef const& ref) const
{
  throw ref;
}

} // namespace o2::framework
