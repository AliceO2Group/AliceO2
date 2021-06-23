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
#ifndef FRAMEWORK_CALLBACKSERVICE_H
#define FRAMEWORK_CALLBACKSERVICE_H

#include "CallbackRegistry.h"
#include "Framework/ServiceHandle.h"
#include <tuple>

class FairMQRegionInfo;

namespace o2
{
namespace framework
{

class EndOfStreamContext;

// A service that data processors can register callback functions invoked by the
// framework at defined steps in the process flow
class CallbackService
{
 public:
  /// Callbacks are a global service because they will always be
  /// invoked by the main thread only.
  constexpr static ServiceKind service_kind = ServiceKind::Global;
  /// the defined processing steps at which a callback can be invoked
  enum class Id {
    Start,     /**< Invoked before the inner loop is started */
    Stop,      /**< Invoked when the device is about to be stoped */
    Reset,     /**< Invoked on device rest */
    Idle,      /**< Invoked when there was no computation scheduled */
    ClockTick, /**< Invoked every iteration of the inner loop */
    /// Invoked when we are notified that no further data will arrive.
    /// Notice that one could have more "EndOfData" notifications. Because
    /// we could be signaled by control that the data flow restarted.
    EndOfStream,

    /// Invoked whenever FairMQ notifies us of a new region
    ///
    /// return AlgorithmSpec::InitCallback{[=](InitContext& ic) {
    ///    auto& callbacks = ic.services().get<CallbackService>();
    ///    callbacks.set(CallbackService::Id::RegionInfoCallback, [](FairMQRegionInfo const& info) {
    ///    ... do GPU init ...
    ///    });
    ///  }
    ///  ...
    ///  return [task](ProcessingContext& pc) {
    ///    // your processing loop. Guaranteed to be called synchronously
    ///    // with the callback
    ///  };
    /// }};
    RegionInfoCallback
  };

  using StartCallback = std::function<void()>;
  using StopCallback = std::function<void()>;
  using ResetCallback = std::function<void()>;
  using IdleCallback = std::function<void()>;
  using ClockTickCallback = std::function<void()>;
  using EndOfStreamCallback = std::function<void(EndOfStreamContext&)>;
  using RegionInfoCallback = std::function<void(FairMQRegionInfo const&)>;

  using Callbacks = CallbackRegistry<Id,                                                          //
                                     RegistryPair<Id, Id::Start, StartCallback>,                  //
                                     RegistryPair<Id, Id::Stop, StopCallback>,                    //
                                     RegistryPair<Id, Id::Reset, ResetCallback>,                  //
                                     RegistryPair<Id, Id::Idle, IdleCallback>,                    //
                                     RegistryPair<Id, Id::ClockTick, ClockTickCallback>,          //
                                     RegistryPair<Id, Id::EndOfStream, EndOfStreamCallback>,      //
                                     RegistryPair<Id, Id::RegionInfoCallback, RegionInfoCallback> //
                                     >;                                                           //

  // set callback for specified processing step
  template <typename U>
  void set(Id id, U&& cb)
  {
    mCallbacks.set(id, std::forward<U>(cb));
  }

  // execute callback for specified processing step with argument pack
  template <typename... TArgs>
  auto operator()(Id id, TArgs&&... args)
  {
    mCallbacks(id, std::forward<TArgs>(args)...);
  }

 private:
  Callbacks mCallbacks;
};

} // namespace framework
} // namespace o2
#endif // FRAMEWORK_CALLBACKSERVICE_H
