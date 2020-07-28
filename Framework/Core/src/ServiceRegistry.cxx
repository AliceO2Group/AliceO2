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
#include "Framework/Tracing.h"
#include "Framework/Logger.h"
#include <iostream>

namespace o2::framework
{

ServiceRegistryBase::ServiceRegistryBase()
{
  mServicesKey.fill(0L);
  mServicesValue.fill(nullptr);
  for (size_t i = 0; i < mServicesBooked.size(); ++i) {
    mServicesBooked[i] = false;
  }
}

/// Type erased service registration. @a typeHash is the
/// hash used to identify the service, @a service is
/// a type erased pointer to the service itself.
/// This method is supposed to be thread safe
void ServiceRegistryBase::registerService(hash_type typeHash, void* service, ServiceKind kind, uint64_t threadId, const char* name) const
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

} // namespace o2::framework
