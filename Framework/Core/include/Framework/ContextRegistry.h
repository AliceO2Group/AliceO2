// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef FRAMEWORK_CONTEXTREGISTRY_H
#define FRAMEWORK_CONTEXTREGISTRY_H

#include "Framework/FairMQDeviceProxy.h"
#include "Framework/DataProcessor.h"

#include <unordered_map>
#include <functional>

namespace o2
{
namespace framework
{

/// Decouples getting the various contextes from the actual type
/// of context, so that the DataAllocator does not need to know
/// about the various serialization methods. Since there is only
/// a few context types we support, this can be done in an ad hoc
/// manner making sure each overload of ContextRegistry<T>::get()
/// uses a different entry in ContextRegistry::contextes;
///
class ContextRegistry
{
 public:
  /// Get a registered context by type. The only requirement
  /// for a context is to have a static bool mRegistered
  /// which needs to be initialised with REGISTER_CONTEXT
  /// macro.
  template <class T, size_t S = sizeof(T)>
  static T* get()
  {
    return reinterpret_cast<T*>(mContextes[&T::mRegistered]);
  }

  /// Default setter to register a given context in the registry.
  /// Notice that since this is templated, Context could actually
  /// overwrite it with their own version.
  template <class T>
  static void set(T* context)
  {
    mContextes[&T::mRegistered] = context;
    mCleaners[&T::mRegistered] = [context]() -> void { context->clear(); };
    mSenders[&T::mRegistered] = [context](FairMQDevice& device) -> void { DataProcessor::doSend(device, *context); };
    mInits[&T::mRegistered] = [context](FairMQDevice* device) -> void { context->mProxy.setDevice(device); };
  }

  template <class T>
  static bool createInRegistry()
  {
    /// We should probably use a unique_ptr<void> and keep track
    /// of the deletion mechanism as well.
    set<T>(new T());
    return true;
  }

  /// Invoke this to clean all the registered Context
  static void clear()
  {
    for (auto & [ _, cleaner ] : mCleaners) {
      cleaner();
    }
  }

  /// Invoke the sender callback on all the registered Context
  static void send(FairMQDevice& device)
  {
    for (auto & [ _, sender ] : mSenders) {
      sender(device);
    }
  }

  /// Invoke the init callback on all the registered Context
  static void init(FairMQDevice* device)
  {
    for (auto & [ _, init ] : mInits) {
      init(device);
    }
  }

 private:
  static inline std::unordered_map<bool*, void*> mContextes = {};
  static inline std::unordered_map<bool*, std::function<void(void)>> mCleaners = {};
  static inline std::unordered_map<bool*, std::function<void(FairMQDevice&)>> mSenders = {};
  static inline std::unordered_map<bool*, std::function<void(FairMQDevice*)>> mInits = {};
};

#define REGISTER_CONTEXT(CONTEXT) \
  friend class ContextRegistry;   \
  static inline bool mRegistered = ContextRegistry::createInRegistry<CONTEXT>();

} // namespace framework
} // namespace o2
#endif // FRAMEWORK_CONTEXTREGISTRY_H
