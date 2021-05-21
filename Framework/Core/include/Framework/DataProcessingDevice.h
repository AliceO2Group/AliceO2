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
#include "Framework/TerminationPolicy.h"
#include "Framework/Tracing.h"
#include "Framework/RunningWorkflowInfo.h"

#include <fairmq/FairMQDevice.h>
#include <fairmq/FairMQParts.h>

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

/// Context associated to a device. In principle
/// multiple DataProcessors can run on a Device (even if we
/// do not do it for now).
struct DeviceContext {
  // These are pointers to the one owned by the DataProcessingDevice
  // and therefore require actual locking
  DataProcessingDevice* device = nullptr;
  DeviceSpec const* spec = nullptr;
  DeviceState* state = nullptr;
  ComputingQuotaEvaluator* quotaEvaluator = nullptr;
  DataProcessingStats* stats = nullptr;
};

struct DataProcessorContext {
  // These are specific of a given context and therefore
  // not shared by threads.
  bool* wasActive = nullptr;
  bool allDone = false;

  // These are pointers to the one owned by the DataProcessingDevice
  // but they are fully reentrant / thread safe and therefore can
  // be accessed without a lock.

  // FIXME: move stuff here from the list below... ;-)
  DeviceContext* deviceContext = nullptr;
  DataRelayer* relayer = nullptr;
  ServiceRegistry* registry = nullptr;
  std::vector<DataRelayer::RecordAction>* completed = nullptr;
  std::vector<ExpirationHandler>* expirationHandlers = nullptr;
  TimingInfo* timingInfo = nullptr;
  DataAllocator* allocator = nullptr;
  AlgorithmSpec::ProcessCallback* statefulProcess = nullptr;
  AlgorithmSpec::ProcessCallback* statelessProcess = nullptr;
  AlgorithmSpec::ErrorCallback* error = nullptr;

  std::function<void(o2::framework::RuntimeErrorRef e, InputRecord& record)>* errorHandling = nullptr;
};

struct TaskStreamRef {
  int index = -1;
};

struct TaskStreamInfo {
  /// The id of this stream
  TaskStreamRef id;
  /// The context of the DataProcessor being run by this task
  DataProcessorContext* context;
  /// Wether or not this task is running
  bool running = false;
};

/// A device actually carrying out all the DPL
/// Data Processing needs.
class DataProcessingDevice : public FairMQDevice
{
 public:
  DataProcessingDevice(RunningDeviceRef ref, ServiceRegistry&);
  void Init() final;
  void InitTask() final;
  void PreRun() final;
  void PostRun() final;
  void Reset() final;
  void ResetTask() final;
  bool ConditionalRun() final;
  void SetErrorPolicy(enum TerminationPolicy policy) { mErrorPolicy = policy; }

  // Processing functions are now renetrant
  static void doRun(DataProcessorContext& context);
  static void doPrepare(DataProcessorContext& context);
  static void handleData(DataProcessorContext& context, FairMQParts&, InputChannelInfo&);
  static bool tryDispatchComputation(DataProcessorContext& context, std::vector<DataRelayer::RecordAction>& completed);
  std::vector<DataProcessorContext> mDataProcessorContexes;

 protected:
  void error(const char* msg);
  void fillContext(DataProcessorContext& context, DeviceContext& deviceContext);

 private:
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
  TimingInfo mTimingInfo;
  DataAllocator mAllocator;
  DataRelayer* mRelayer = nullptr;
  /// Expiration handler
  std::vector<ExpirationHandler> mExpirationHandlers;
  /// Completed actions
  std::vector<DataRelayer::RecordAction> mCompleted;

  uint64_t mLastSlowMetricSentTimestamp = 0;         /// The timestamp of the last time we sent slow metrics
  uint64_t mLastMetricFlushedTimestamp = 0;          /// The timestamp of the last time we actually flushed metrics
  uint64_t mBeginIterationTimestamp = 0;             /// The timestamp of when the current ConditionalRun was started
  DataProcessingStats mStats;                        /// Stats about the actual data processing.
  std::vector<FairMQRegionInfo> mPendingRegionInfos; /// A list of the region infos not yet notified.
  std::mutex mRegionInfoMutex;
  enum TerminationPolicy mErrorPolicy = TerminationPolicy::WAIT; /// What to do when an error arises
  bool mWasActive = false;                                       /// Whether or not the device was active at last iteration.
  std::vector<uv_work_t> mHandles;                               /// Handles to use to schedule work.
  std::vector<TaskStreamInfo> mStreams;                          /// Information about the task running in the associated mHandle.
  ComputingQuotaEvaluator& mQuotaEvaluator;                      /// The component which evaluates if the offer can be used to run a task
};

} // namespace o2::framework
#endif
