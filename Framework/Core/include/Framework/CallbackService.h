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
#ifndef O2_FRAMEWORK_CALLBACKSERVICE_H_
#define O2_FRAMEWORK_CALLBACKSERVICE_H_

#include "CallbackRegistry.h"
#include "Framework/ServiceHandle.h"
#include "Framework/DataProcessingHeader.h"
#include "ServiceRegistry.h"

#include <fairmq/FwdDecls.h>

namespace o2::header
{
struct DataHeader;
}

namespace o2::framework
{

struct ConcreteDataMatcher;
struct EndOfStreamContext;

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
    Start,        /**< Invoked before the inner loop is started */
    Stop,         /**< Invoked when the device is about to be stoped */
    Reset,        /**< Invoked on device rest */
    Idle,         /**< Invoked when there was no computation scheduled */
    ClockTick,    /**< Invoked every iteration of the inner loop */
    DataConsumed, /**< Invoked whenever data has been consumed */
    /// Invoked when we are notified that no further data will arrive.
    /// Notice that one could have more "EndOfData" notifications. Because
    /// we could be signaled by control that the data flow restarted.
    EndOfStream,

    /// Invoked whenever FairMQ notifies us of a new region
    ///
    /// return AlgorithmSpec::InitCallback{[=](InitContext& ic) {
    ///    auto& callbacks = ic.services().get<CallbackService>();
    ///    callbacks.set(CallbackService::Id::RegionInfoCallback, [](fair::mq::RegionInfo const& info) {
    ///    ... do GPU init ...
    ///    });
    ///  }
    ///  ...
    ///  return [task](ProcessingContext& pc) {
    ///    // your processing loop. Guaranteed to be called synchronously
    ///    // with the callback
    ///  };
    /// }};
    RegionInfoCallback,
    /// Invoked whenever a new timeslice has been created from an enumeration.
    /// Users can override this to make sure the fill the DataHeader associated
    /// to a timeslice with the wanted quantities.
    NewTimeslice,
    /// Invoked before the processing callback
    PreProcessing,
    /// Invoked after the processing callback,
    PostProcessing,
    /// Invoked whenever an object from CCDB is deserialised via ROOT.
    /// Use this to finalise the initialisation of the object.
    CCDBDeserialised,
    /// Invoked when new domain info is available
    DomainInfoUpdated,
    /// Invoked the device undergoes a state change
    DeviceStateChanged,
    /// Invoked when the device was requested to stop
    ExitRequested,
  };

  using StartCallback = std::function<void()>;
  using StopCallback = std::function<void()>;
  using ResetCallback = std::function<void()>;
  using IdleCallback = std::function<void()>;
  using ClockTickCallback = std::function<void()>;
  using DataConsumedCallback = std::function<void(ServiceRegistryRef)>;
  using EndOfStreamCallback = std::function<void(EndOfStreamContext&)>;
  using RegionInfoCallback = std::function<void(fair::mq::RegionInfo const&)>;
  using NewTimesliceCallback = std::function<void(o2::header::DataHeader&, DataProcessingHeader&)>;
  using PreProcessingCallback = std::function<void(ServiceRegistryRef, int)>;
  using PostProcessingCallback = std::function<void(ServiceRegistryRef, int)>;
  using CCDBDeserializedCallback = std::function<void(ConcreteDataMatcher&, void*)>;
  using DomainInfoUpdatedCallback = std::function<void(ServiceRegistryRef, size_t timeslice, ChannelIndex index)>;
  using DeviceStateChangedCallback = std::function<void(ServiceRegistryRef, int newState)>;
  using ExitRequestedCallback = std::function<void(ServiceRegistryRef)>;

  using Callbacks = CallbackRegistry<Id,                                                                   //
                                     RegistryPair<Id, Id::Start, StartCallback>,                           //
                                     RegistryPair<Id, Id::Stop, StopCallback>,                             //
                                     RegistryPair<Id, Id::Reset, ResetCallback>,                           //
                                     RegistryPair<Id, Id::Idle, IdleCallback>,                             //
                                     RegistryPair<Id, Id::ClockTick, ClockTickCallback>,                   //
                                     RegistryPair<Id, Id::DataConsumed, DataConsumedCallback>,             //
                                     RegistryPair<Id, Id::EndOfStream, EndOfStreamCallback>,               //
                                     RegistryPair<Id, Id::RegionInfoCallback, RegionInfoCallback>,         //
                                     RegistryPair<Id, Id::NewTimeslice, NewTimesliceCallback>,             //
                                     RegistryPair<Id, Id::PreProcessing, PreProcessingCallback>,           //
                                     RegistryPair<Id, Id::PostProcessing, PostProcessingCallback>,         //
                                     RegistryPair<Id, Id::CCDBDeserialised, CCDBDeserializedCallback>,     //
                                     RegistryPair<Id, Id::DomainInfoUpdated, DomainInfoUpdatedCallback>,   //
                                     RegistryPair<Id, Id::DeviceStateChanged, DeviceStateChangedCallback>, //
                                     RegistryPair<Id, Id::ExitRequested, ExitRequestedCallback>            //
                                     >;                                                                    //

  // Typesafe API to register callbacks
  template <Id id, typename U>
  void set(U&& callback)
  {
    mCallbacks.set(id, std::forward<U>(callback));
  }

  // Typesafe API to invoke callbacks
  template <Id id, typename... TArgs>
  auto call(TArgs&&... args)
  {
    mCallbacks(id, std::forward<TArgs>(args)...);
  }

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

} // namespace o2::framework
#endif // O2_FRAMEWORK_CALLBACKSERVICE_H_
