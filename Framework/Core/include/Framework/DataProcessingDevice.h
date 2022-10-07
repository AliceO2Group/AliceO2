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
#ifndef O2_FRAMEWORK_DATAPROCESSINGDEVICE_H_
#define O2_FRAMEWORK_DATAPROCESSINGDEVICE_H_

#include "Framework/AlgorithmSpec.h"
#include "Framework/ComputingQuotaOffer.h"
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
#include "Framework/ProcessingPolicies.h"
#include "Framework/Tracing.h"
#include "Framework/RunningWorkflowInfo.h"
#include "Framework/ObjectCache.h"
#include "Framework/DataProcessingContext.h"
#include "Framework/DeviceContext.h"

#include <fairmq/Device.h>
#include <fairmq/Parts.h>

#include <memory>
#include <mutex>
#include <uv.h>

namespace o2::framework
{

struct InputChannelInfo;
struct DeviceState;
struct ComputingQuotaEvaluator;

/// Context associated to a given DataProcessor.
/// For the time being everything points to
/// members of the DataProcessingDevice and
/// we make sure that things are serialised by
/// locking a global lock. We can then have per
/// thread instances for what makes sense to have
/// per thread and relax the locks.
class DataProcessingDevice;

struct TaskStreamRef {
  int index = -1;
};

struct TaskStreamInfo {
  /// The id of this stream
  TaskStreamRef id;
  /// The context of the DataProcessor being run by this task
  DataProcessorContext* context;
  /// The libuv task handle
  uv_work_t task;
  /// Wether or not this task is running
  bool running = false;
};

struct DeviceConfigurationHelpers {
  static std::unique_ptr<ConfigParamStore> getConfiguration(ServiceRegistryRef registry, const char* name, std::vector<ConfigParamSpec> const& options);
};

/// A device actually carrying out all the DPL
/// Data Processing needs.
class DataProcessingDevice : public fair::mq::Device
{
 public:
  DataProcessingDevice(RunningDeviceRef ref, ServiceRegistry&, ProcessingPolicies& policies);
  void Init() final;
  void InitTask() final;
  void PreRun() final;
  void PostRun() final;
  void Reset() final;
  void ResetTask() final;
  void Run() final;

  // Processing functions are now renetrant
  static void doRun(DataProcessorContext& context);
  static void doPrepare(DataProcessorContext& context);
  static void handleData(DataProcessorContext& context, InputChannelInfo&);
  static bool tryDispatchComputation(DataProcessorContext& context, std::vector<DataRelayer::RecordAction>& completed);
  std::vector<DataProcessorContext> mDataProcessorContexes;

 protected:
  void error(const char* msg);
  void fillContext(DataProcessorContext& context, DeviceContext& deviceContext);

 private:
  /// Initialise the socket pollers / timers
  void initPollers();
  void startPollers();
  void stopPollers();
  DeviceContext mDeviceContext;
  /// The specification used to create the initial state of this device
  DeviceSpec const& mSpec;
  /// The current internal state of this device.
  DeviceState& mState;

  AlgorithmSpec::InitCallback mInit;
  AlgorithmSpec::ProcessCallback mStatefulProcess;
  AlgorithmSpec::ProcessCallback mStatelessProcess;
  AlgorithmSpec::ErrorCallback mError;
  std::function<void(RuntimeErrorRef e, InputRecord& record)> mErrorHandling;
  std::unique_ptr<ConfigParamRegistry> mConfigRegistry;
  ServiceRegistry& mServiceRegistry;
  DataAllocator mAllocator;
  /// Expiration handler
  std::vector<ExpirationHandler> mExpirationHandlers;
  /// Completed actions
  std::vector<DataRelayer::RecordAction> mCompleted;

  uint64_t mLastSlowMetricSentTimestamp = 0;         /// The timestamp of the last time we sent slow metrics
  uint64_t mLastMetricFlushedTimestamp = 0;          /// The timestamp of the last time we actually flushed metrics
  uint64_t mBeginIterationTimestamp = 0;             /// The timestamp of when the current ConditionalRun was started
  DataProcessingStats mStats;                        /// Stats about the actual data processing.
  std::vector<fair::mq::RegionInfo> mPendingRegionInfos; /// A list of the region infos not yet notified.
  std::mutex mRegionInfoMutex;
  ProcessingPolicies mProcessingPolicies;                        /// User policies related to data processing
  bool mWasActive = false;                                       /// Whether or not the device was active at last iteration.
  std::vector<uv_work_t> mHandles;                               /// Handles to use to schedule work.
  std::vector<TaskStreamInfo> mStreams;                          /// Information about the task running in the associated mHandle.
  ComputingQuotaEvaluator& mQuotaEvaluator;                      /// The component which evaluates if the offer can be used to run a task
  /// Handle to wake up the main loop from other threads
  /// e.g. when FairMQ notifies some callback in an asynchronous way
  uv_async_t* mAwakeHandle = nullptr;
};

} // namespace o2::framework
#endif // O2_FRAMEWORK_DATAPROCESSINGDEVICE_H_
