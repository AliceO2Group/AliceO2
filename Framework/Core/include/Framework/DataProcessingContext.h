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
#ifndef O2_FRAMEWORK_DATAPROCESSORCONTEXT_H_
#define O2_FRAMEWORK_DATAPROCESSORCONTEXT_H_

#include "Framework/DataRelayer.h"
#include "Framework/AlgorithmSpec.h"
#include <functional>

namespace o2::framework
{

struct DeviceContext;
struct ServiceRegistry;
struct DataAllocator;
struct DataProcessorSpec;

struct DataProcessorContext {
  DataProcessorContext(DataProcessorContext const&) = delete;
  DataProcessorContext() = default;
  // These are specific of a given context and therefore
  // not shared by threads.
  bool* wasActive = nullptr;
  bool allDone = false;
  /// Latest run number we processed globally for this DataProcessor.
  int64_t lastRunNumberProcessed = 0;

  // These are pointers to the one owned by the DataProcessingDevice
  // but they are fully reentrant / thread safe and therefore can
  // be accessed without a lock.

  // FIXME: move stuff here from the list below... ;-)
  ServiceRegistry* registry = nullptr;
  std::vector<DataRelayer::RecordAction> completed;
  std::vector<ExpirationHandler> expirationHandlers;
  AlgorithmSpec::InitCallback init;
  AlgorithmSpec::ProcessCallback statefulProcess;
  AlgorithmSpec::ProcessCallback statelessProcess;
  AlgorithmSpec::ErrorCallback error = nullptr;

  DataProcessorSpec* spec = nullptr; /// Invoke callbacks to be executed in PreRun(), before the User Start callbacks

  /// Invoke callbacks to be executed before starting the processing loop
  void preStartCallbacks(ServiceRegistryRef);
  /// Invoke callbacks to be executed before every process method invokation
  void preProcessingCallbacks(ProcessingContext&);
  /// Invoke callbacks to be executed after every process method invokation
  void postProcessingCallbacks(ProcessingContext&);
  /// Invoke callbacks to be executed before every dangling check
  void preDanglingCallbacks(DanglingContext&);
  /// Invoke callbacks to be executed after every dangling check
  void postDanglingCallbacks(DanglingContext&);
  /// Invoke callbacks to be executed before every EOS user callback invokation
  void preEOSCallbacks(EndOfStreamContext&);
  /// Invoke callbacks to be executed after every EOS user callback invokation
  void postEOSCallbacks(EndOfStreamContext&);
  /// Invoke callbacks to monitor inputs after dispatching, regardless of them
  /// being discarded, consumed or processed.
  void postDispatchingCallbacks(ProcessingContext&);
  /// Callback invoked after the late forwarding has been done
  void postForwardingCallbacks(ProcessingContext&);

  /// Invoke callbacks on stop.
  void postStopCallbacks(ServiceRegistryRef);
  /// Invoke callbacks on exit.
  /// Note how this is a static helper because otherwise we would need to
  /// handle differently the deletion of the DataProcessingContext itself.
  static void preExitCallbacks(std::vector<ServiceExitHandle>, ServiceRegistryRef);

  /// Invoke whenever we get a new DomainInfo message
  void domainInfoUpdatedCallback(ServiceRegistryRef, size_t oldestPossibleTimeslice, ChannelIndex channelIndex);

  /// Invoke before sending messages @a parts on a channel @a channelindex
  void preSendingMessagesCallbacks(ServiceRegistryRef, fair::mq::Parts& parts, ChannelIndex channelindex);

  /// Callback for services to be executed before every processing.
  /// The callback MUST BE REENTRANT and threadsafe.
  mutable std::vector<ServiceProcessingHandle> preProcessingHandlers;
  /// Callback for services to be executed after every processing.
  /// The callback MUST BE REENTRANT and threadsafe.
  mutable std::vector<ServiceProcessingHandle> postProcessingHandlers;
  /// Callbacks for services to be executed before every dangling check
  mutable std::vector<ServiceDanglingHandle> preDanglingHandles;
  /// Callbacks for services to be executed after every dangling check
  mutable std::vector<ServiceDanglingHandle> postDanglingHandles;
  /// Callbacks for services to be executed before every EOS user callback invokation
  mutable std::vector<ServiceEOSHandle> preEOSHandles;
  /// Callbacks for services to be executed after every EOS user callback invokation
  mutable std::vector<ServiceEOSHandle> postEOSHandles;
  /// Callbacks for services to be executed after every dispatching
  mutable std::vector<ServiceDispatchingHandle> postDispatchingHandles;
  /// Callbacks for services to be executed after every dispatching
  mutable std::vector<ServiceForwardingHandle> postForwardingHandles;
  /// Callbacks for services to be executed before Start
  mutable std::vector<ServiceStartHandle> preStartHandles;
  /// Callbacks for services to be executed on the Stop transition
  mutable std::vector<ServiceStopHandle> postStopHandles;
  /// Callbacks for services to be executed on exit
  mutable std::vector<ServiceExitHandle> preExitHandles;
  /// Callbacks for services to be executed on exit
  mutable std::vector<ServiceDomainInfoHandle> domainInfoHandles;
  /// Callbacks for services to be executed before sending messages
  mutable std::vector<ServicePreSendingMessagesHandle> preSendingMessagesHandles;

  /// Wether or not the associated DataProcessor can forward things early
  bool canForwardEarly = true;
  bool isSink = false;
  bool balancingInputs = true;

  std::function<void(o2::framework::RuntimeErrorRef e, InputRecord& record)> errorHandling;
};

} // namespace o2::framework
#endif // O2_FRAMEWORK_DATAPROCESSINGCONTEXT_H_
