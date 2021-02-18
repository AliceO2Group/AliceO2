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
#include "Framework/DataAllocator.h"
#include "Framework/DataRelayer.h"
#include "Framework/DeviceSpec.h"
#include "Framework/DataProcessingStats.h"
#include "Framework/ExpirationHandler.h"
#include "Framework/ServiceRegistry.h"
#include "Framework/InputRoute.h"
#include "Framework/ForwardRoute.h"
#include "Framework/TimingInfo.h"
#include "Framework/TerminationPolicy.h"

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
  void InitTask() final;
  void PreRun() final;
  void PostRun() final;
  void Reset() final;
  void ResetTask() final;
  bool ConditionalRun() final;
  void SetErrorPolicy(enum TerminationPolicy policy) { mErrorPolicy = policy; }
  void bindService(ServiceSpec const& spec, void* service);

 protected:
  bool doRun();
  bool handleData(FairMQParts&, InputChannelInfo&);
  bool tryDispatchComputation(std::vector<DataRelayer::RecordAction>& completed);
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
  std::function<void(std::exception& e, InputRecord& record)> mErrorHandling;
  std::unique_ptr<ConfigParamRegistry> mConfigRegistry;
  ServiceRegistry& mServiceRegistry;
  TimingInfo mTimingInfo;
  DataAllocator mAllocator;
  DataRelayer* mRelayer = nullptr;
  std::vector<ExpirationHandler> mExpirationHandlers;
  std::vector<DataRelayer::RecordAction> mCompleted;
  /// Callbacks for services to be executed before every process method invokation
  std::vector<ServiceProcessingHandle> mPreProcessingHandles;
  /// Callbacks for services to be executed after every process method invokation
  std::vector<ServiceProcessingHandle> mPostProcessingHandles;
  /// Callbacks for services to be executed before every EOS user callback invokation
  std::vector<ServiceEOSHandle> mPreEOSHandles;
  /// Callbacks for services to be executed after every EOS user callback invokation
  std::vector<ServiceEOSHandle> mPostEOSHandles;

  int mErrorCount;
  uint64_t mLastSlowMetricSentTimestamp = 0;         /// The timestamp of the last time we sent slow metrics
  uint64_t mLastMetricFlushedTimestamp = 0;          /// The timestamp of the last time we actually flushed metrics
  uint64_t mBeginIterationTimestamp = 0;             /// The timestamp of when the current ConditionalRun was started
  DataProcessingStats mStats;                        /// Stats about the actual data processing.
  std::vector<FairMQRegionInfo> mPendingRegionInfos; /// A list of the region infos not yet notified.
  enum TerminationPolicy mErrorPolicy = TerminationPolicy::WAIT; /// What to do when an error arises
  bool mWasActive = false;                                       /// Whether or not the device was active at last iteration.
};

} // namespace o2::framework
#endif
