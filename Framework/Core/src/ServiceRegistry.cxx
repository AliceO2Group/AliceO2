// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#include "Framework/ServiceRegistry.h"
#include "Framework/Logger.h"
#include <iostream>

namespace o2::framework
{

ServiceRegistry::ServiceRegistry()
{
  for (size_t i = 0; i < MAX_SERVICES; ++i) {
    mServicesKey[i].store(0LL);
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
      std::atomic_thread_fence(std::memory_order_release);
      mServicesKey[i + threadHashId] = typeHash;
      return;
    }
  }
  throw std::runtime_error(std::string("Unable to find a spot in the registry for service ") +
                           std::to_string(typeHash) +
                           ". Make sure you use const / non-const correctly.");
}

void ServiceRegistry::declareService(ServiceSpec const& spec)
{
  mSpecs.push_back(spec);
  // Services which are not stream must have a single instance created upfront.
  if (spec.kind != ServiceKind::Stream) {
    ServiceHandle handle = spec.init(*this, *mState, *mOptions);
    this->registerService(handle.hash, handle.instance, handle.kind, 0, handle.name.c_str());
    this->bindService(spec, handle.instance);
  }
}

void ServiceRegistry::bindState(DeviceState* state, fair::mq::ProgOptions* options)
{
  mState = state;
  mOptions = options;
}

void ServiceRegistry::bindService(ServiceSpec const& spec, void* service)
{
  static TracyLockableN(std::mutex, checkIfPresent, "service method binding mutex");
  static std::set<std::string> boundServices; 
  {
    std::scoped_lock<LockableBase(std::mutex)> lock(checkIfPresent);
    if (boundServices.count(spec.name)) {
      return;
    }
    boundServices.insert(spec.name);
  }
  int nextPos = mNextServiceSlot++;
  mPreProcessingHandles[nextPos] = ServiceProcessingHandle{spec.preProcessing};
  mPostProcessingHandles[nextPos] = ServiceProcessingHandle{spec.postProcessing};
  mPreDanglingHandles[nextPos] = ServiceDanglingHandle{spec.preDangling};
  mPostDanglingHandles[nextPos] = ServiceDanglingHandle{spec.postDangling};
  mPreEOSHandles[nextPos] = ServiceEOSHandle{spec.preEOS};
  mPostEOSHandles[nextPos] = ServiceEOSHandle{spec.postEOS};
  // Spinlock to serialise the insertion, since it's done rarely.
  while (!mBoundServices.compare_exchange_strong(nextPos, nextPos + 1)) {
  };
}

/// Invoke callbacks to be executed before every process method invokation
void ServiceRegistry::preProcessingCallbacks(ProcessingContext& processContext)
{
  for (size_t i = 0; i < mBoundServices.load(); ++i) {
    auto& handle = mPreProcessingHandles[i];
    if (handle.callback) {
      handle.callback(processContext);
    }
  }
}
/// Invoke callbacks to be executed after every process method invokation
void ServiceRegistry::postProcessingCallbacks(ProcessingContext& processContext)
{
  for (size_t i = 0; i < mBoundServices.load(); ++i) {
    auto& handle = mPostProcessingHandles[i];
    if (handle.callback) {
      handle.callback(processContext);
    }
  }
}
/// Invoke callbacks to be executed before every dangling check
void ServiceRegistry::preDanglingCallbacks(DanglingContext& danglingContext)
{
  for (size_t i = 0; i < mBoundServices.load(); ++i) {
    auto& preDanglingHandle = mPreDanglingHandles[i];
    if (preDanglingHandle.callback) {
      preDanglingHandle.callback(danglingContext);
    }
  }
}

/// Invoke callbacks to be executed after every dangling check
void ServiceRegistry::postDanglingCallbacks(DanglingContext& danglingContext)
{
  for (size_t i = 0; i < mBoundServices.load(); ++i) {
    auto& postDanglingHandle = mPostDanglingHandles[i];
    if (postDanglingHandle.callback) {
      postDanglingHandle.callback(danglingContext);
    }
  }
}

/// Invoke callbacks to be executed before every EOS user callback invokation
void ServiceRegistry::preEOSCallbacks(EndOfStreamContext& eosContext)
{
  for (size_t i = 0; i < mBoundServices.load(); ++i) {
    auto& eosHandle = mPreEOSHandles[i];
    if (eosHandle.callback) {
      eosHandle.callback(eosContext);
    }
  }
}

/// Invoke callbacks to be executed after every EOS user callback invokation
void ServiceRegistry::postEOSCallbacks(EndOfStreamContext& eosContext)
{
  for (size_t i = 0; i < mBoundServices.load(); ++i) {
    auto& eosHandle = mPostEOSHandles[i];
    if (eosHandle.callback) {
      eosHandle.callback(eosContext);
    }
  }
}

} // namespace o2::framework
