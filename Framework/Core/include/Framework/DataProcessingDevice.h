// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#ifndef FRAMEWORK_DATAPROCESSING_DEVICE_H
#define FRAMEWORK_DATAPROCESSING_DEVICE_H

#include "Framework/AlgorithmSpec.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/ContextRegistry.h"
#include "Framework/DataAllocator.h"
#include "Framework/DataRelayer.h"
#include "Framework/DeviceSpec.h"
#include "Framework/DataProcessingStats.h"
#include "Framework/ExpirationHandler.h"
#include "Framework/MessageContext.h"
#include "Framework/RootObjectContext.h"
#include "Framework/ArrowContext.h"
#include "Framework/StringContext.h"
#include "Framework/RawBufferContext.h"
#include "Framework/ServiceRegistry.h"
#include "Framework/InputRoute.h"
#include "Framework/ForwardRoute.h"
#include "Framework/TimingInfo.h"

#include <fairmq/FairMQDevice.h>
#include <fairmq/FairMQParts.h>

#include <memory>

namespace o2::framework
{

struct InputChannelInfo;
struct DeviceState;

/// A device actually carrying out all the DPL
/// Data Processing needs.
class DataProcessingDevice : public FairMQDevice
{
 public:
  DataProcessingDevice(DeviceSpec const& spec, ServiceRegistry&, DeviceState& state);
  void Init() final;
  void PreRun() final;
  void PostRun() final;
  void Reset() final;
  bool ConditionalRun() final;

 protected:
  bool handleData(FairMQParts&, InputChannelInfo&);
  bool tryDispatchComputation();
  void error(const char* msg);

 private:
  /// The specification used to create the initial state of this device
  DeviceSpec const& mSpec;
  /// The current internal state of this device.
  DeviceState& mState;
  AlgorithmSpec::InitCallback mInit;
  AlgorithmSpec::ProcessCallback mStatefulProcess;
  AlgorithmSpec::ProcessCallback mStatelessProcess;
  AlgorithmSpec::ErrorCallback mError;
  std::unique_ptr<ConfigParamRegistry> mConfigRegistry;
  ServiceRegistry& mServiceRegistry;
  TimingInfo mTimingInfo;
  MessageContext mFairMQContext;
  RootObjectContext mRootContext;
  StringContext mStringContext;
  ArrowContext mDataFrameContext;
  RawBufferContext mRawBufferContext;
  ContextRegistry mContextRegistry;
  DataAllocator mAllocator;
  DataRelayer mRelayer;
  std::vector<ExpirationHandler> mExpirationHandlers;

  int mErrorCount;
  int mProcessingCount;
  uint64_t mLastSlowMetricSentTimestamp = 0; /// The timestamp of the last time we sent slow metrics
  uint64_t mLastMetricFlushedTimestamp = 0;  /// The timestamp of the last time we actually flushed metrics
  uint64_t mBeginIterationTimestamp = 0;     /// The timestamp of when the current ConditionalRun was started
  DataProcessingStats mStats;                /// Stats about the actual data processing.
};

} // namespace o2::framework
#endif
