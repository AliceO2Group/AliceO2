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

#include "Framework/ServiceHandle.h"
#include "Framework/DataProcessingHeader.h"
#include "Framework/ServiceRegistryRef.h"
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
    DeviceStateChanged
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
  using DeviceStateChangedCallback = std::function<void(ServiceRegistryRef, int newState)>;

  using CCDBDeserializedCallback = std::function<void(ConcreteDataMatcher&, void*)>;

  using DomainInfoUpdatedCallback = std::function<void(ServiceRegistryRef, size_t timeslice, ChannelIndex index)>;

  // set callback for specified processing step
  template <typename U>
  void set(Id id, U&& cb)
  {
    auto f = std::function(std::forward<U>(cb));
    using T = std::decay_t<decltype(f)>;
    if constexpr (std::is_same_v<T, StartCallback>) {
      switch (id) {
        case Id::Start:
          mStartCallback = f;
          break;
        case Id::Stop:
          mStopCallback = f;
          break;
        case Id::Reset:
          mResetCallback = f;
          break;
        case Id::Idle:
          mIdleCallback = f;
          break;
        case Id::ClockTick:
          mClockTickCallback = f;
          break;
        default:
          throw std::runtime_error("Invalid callback type");
      }
    } else if constexpr (std::is_same_v<T, DataConsumedCallback>) {
      mDataConsumedCallback = f;
    } else if constexpr (std::is_same_v<T, EndOfStreamCallback>) {
      mEndOfStreamCallback = f;
    } else if constexpr (std::is_same_v<T, RegionInfoCallback>) {
      mRegionInfoCallback = f;
    } else if constexpr (std::is_same_v<T, NewTimesliceCallback>) {
      mNewTimesliceCallback = f;
    } else if constexpr (std::is_same_v<T, PreProcessingCallback>) {
      switch (id) {
        case Id::PreProcessing:
          mPreProcessingCallback = f;
          break;
        case Id::PostProcessing:
          mPostProcessingCallback = f;
          break;
        case Id::DeviceStateChanged:
          mDeviceStateChangedCallback = f;
          break;
        default:
          throw std::runtime_error("Invalid callback type");
      }
    } else if constexpr (std::is_same_v<T, CCDBDeserializedCallback>) {
      mCCDBDeserializedCallback = f;
    } else if constexpr (std::is_same_v<T, DomainInfoUpdatedCallback>) {
      mDomainInfoUpdatedCallback = f;
    } else {
      static_assert(always_static_assert_v<T>, "Unsupported callback type");
    }
  }

  template <typename... TArgs>
  void operator()(Id id, TArgs&&... args)
  {
    using T = std::function<void(TArgs...)>;
    if constexpr (std::is_same_v<T, StartCallback>) {
      switch (id) {
        case Id::Start:
          if (mStartCallback) {
            mStartCallback(std::forward<TArgs>(args)...);
          }
          break;
        case Id::Stop:
          if (mStopCallback) {
            mStopCallback(std::forward<TArgs>(args)...);
          }
          break;
        case Id::Reset:
          if (mResetCallback) {
            mResetCallback(std::forward<TArgs>(args)...);
          }
          break;
        case Id::Idle:
          if (mIdleCallback) {
            mIdleCallback(std::forward<TArgs>(args)...);
          }
          break;
        case Id::ClockTick:
          if (mClockTickCallback) {
            mClockTickCallback(std::forward<TArgs>(args)...);
          }
          break;
        default:
          throw std::runtime_error("Invalid callback type");
      }
    } else if constexpr (std::is_same_v<T, DataConsumedCallback>) {
      if (mDataConsumedCallback) {
        mDataConsumedCallback(std::forward<TArgs>(args)...);
      }
    } else if constexpr (std::is_same_v<T, EndOfStreamCallback>) {
      if (mEndOfStreamCallback) {
        mEndOfStreamCallback(std::forward<TArgs>(args)...);
      }
    } else if constexpr (std::is_same_v<T, RegionInfoCallback>) {
      if (mRegionInfoCallback) {
        mRegionInfoCallback(std::forward<TArgs>(args)...);
      }
    } else if constexpr (std::is_same_v<T, NewTimesliceCallback>) {
      if (mNewTimesliceCallback) {
        mNewTimesliceCallback(std::forward<TArgs>(args)...);
      }
    } else if constexpr (std::is_same_v<T, PreProcessingCallback>) {
      switch (id) {
        case Id::PreProcessing:
          if (mPreProcessingCallback) {
            mPreProcessingCallback(std::forward<TArgs>(args)...);
          }
          break;
        case Id::PostProcessing:
          if (mPostProcessingCallback) {
            mPostProcessingCallback(std::forward<TArgs>(args)...);
          }
          break;
        case Id::DeviceStateChanged:
          if (mDeviceStateChangedCallback) {
            mDeviceStateChangedCallback(std::forward<TArgs>(args)...);
          }
          break;
        default:
          throw std::runtime_error("Invalid callback type");
      }
    } else if constexpr (std::is_same_v<T, CCDBDeserializedCallback>) {
      if (mCCDBDeserializedCallback) {
        mCCDBDeserializedCallback(std::forward<TArgs>(args)...);
      }
    } else if constexpr (std::is_same_v<T, DomainInfoUpdatedCallback>) {
      if (mDomainInfoUpdatedCallback) {
        mDomainInfoUpdatedCallback(std::forward<TArgs>(args)...);
      }
    } else {
      static_assert(always_static_assert_v<T>, "Unsupported callback type");
    }
  }

 private:
  StartCallback mStartCallback;
  StopCallback mStopCallback;
  ResetCallback mResetCallback;
  IdleCallback mIdleCallback;
  ClockTickCallback mClockTickCallback;
  DataConsumedCallback mDataConsumedCallback;
  EndOfStreamCallback mEndOfStreamCallback;
  RegionInfoCallback mRegionInfoCallback;
  NewTimesliceCallback mNewTimesliceCallback;
  PreProcessingCallback mPreProcessingCallback;
  PostProcessingCallback mPostProcessingCallback;
  CCDBDeserializedCallback mCCDBDeserializedCallback;
  DomainInfoUpdatedCallback mDomainInfoUpdatedCallback;
  DeviceStateChangedCallback mDeviceStateChangedCallback;
};

} // namespace o2::framework
#endif // O2_FRAMEWORK_CALLBACKSERVICE_H_
