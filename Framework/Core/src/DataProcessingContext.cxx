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

#include "Framework/DataProcessingContext.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/Signpost.h"

O2_DECLARE_DYNAMIC_LOG(data_processor_context);
namespace o2::framework
{

namespace
{
template <typename T, typename... ARGS>
void invokeAll(T& handles, char const* callbackName, o2::framework::DataProcessorSpec* spec, ARGS&... args)
{
  assert(callbackName);
  O2_SIGNPOST_ID_FROM_POINTER(dpid, data_processor_context, spec);
  // FIXME: for now spec is nullptr because we don't have a list of possible DataProcessorSpecs
  // per device.
  char const* dataProcessorName = spec ? spec->name.c_str() : "DataProcessorContext";
  O2_SIGNPOST_START(data_processor_context, dpid, "callbacks", "Starting %{public}s::%{public}s", dataProcessorName, callbackName);
  for (auto& handle : handles) {
    O2_SIGNPOST_ID_FROM_POINTER(cid, data_processor_context, handle.service);
    O2_SIGNPOST_START(data_processor_context, cid, "callbacks", "Starting %{public}s::%{public}s::%{public}s", dataProcessorName, handle.spec.name.c_str(), callbackName);
    handle.callback(args..., handle.service);
    O2_SIGNPOST_END(data_processor_context, cid, "callbacks", "Ending %{public}s::%{public}s::%{public}s", dataProcessorName, handle.spec.name.c_str(), callbackName);
  }
  O2_SIGNPOST_END(data_processor_context, dpid, "callbacks", "Ending %{public}s::%{public}s", dataProcessorName, callbackName);
}
} // namespace

/// Invoke callbacks to be executed before every dangling check
void DataProcessorContext::preProcessingCallbacks(ProcessingContext& ctx)
{
  invokeAll(preProcessingHandlers, "preProcessingCallbacks", spec, ctx);
}

void DataProcessorContext::finaliseOutputsCallbacks(ProcessingContext& ctx)
{
  invokeAll(finaliseOutputsHandles, "finaliseOutputsCallbacks", spec, ctx);
}

/// Invoke callbacks to be executed before every dangling check
void DataProcessorContext::postProcessingCallbacks(ProcessingContext& ctx)
{
  invokeAll(postProcessingHandlers, "postProcessingCallbacks", spec, ctx);
}

/// Invoke callbacks to be executed before every dangling check
void DataProcessorContext::preDanglingCallbacks(DanglingContext& ctx)
{
  invokeAll(preDanglingHandles, "preDanglingCallbacks", spec, ctx);
}

/// Invoke callbacks to be executed after every dangling check
void DataProcessorContext::postDanglingCallbacks(DanglingContext& ctx)
{
  invokeAll(postDanglingHandles, "postDanglingCallbacks", spec, ctx);
}

/// Invoke callbacks to be executed before every EOS user callback invokation
void DataProcessorContext::preEOSCallbacks(EndOfStreamContext& ctx)
{
  invokeAll(preEOSHandles, "preEOSCallbacks", spec, ctx);
}

/// Invoke callbacks to be executed after every EOS user callback invokation
void DataProcessorContext::postEOSCallbacks(EndOfStreamContext& ctx)
{
  invokeAll(postEOSHandles, "postEOSCallbacks", spec, ctx);
}

/// Invoke callbacks to be executed after every data Dispatching
void DataProcessorContext::postDispatchingCallbacks(ProcessingContext& ctx)
{
  invokeAll(postDispatchingHandles, "postDispatchingCallbacks", spec, ctx);
}

/// Invoke callbacks to be executed after every data Dispatching
void DataProcessorContext::postForwardingCallbacks(ProcessingContext& ctx)
{
  invokeAll(postForwardingHandles, "postForwardingCallbacks", spec, ctx);
}

/// Callbacks to be called in fair::mq::Device::PreRun()
void DataProcessorContext::preStartCallbacks(ServiceRegistryRef ref)
{
  invokeAll(preStartHandles, "preStartCallbacks", spec, ref);
}

void DataProcessorContext::postStopCallbacks(ServiceRegistryRef ref)
{
  invokeAll(postStopHandles, "postStopCallbacks", spec, ref);
}

/// Invoke callback to be executed on exit, in reverse order.
void DataProcessorContext::preExitCallbacks(std::vector<ServiceExitHandle> handles, ServiceRegistryRef ref)
{
  O2_SIGNPOST_ID_FROM_POINTER(dpid, data_processor_context, &ref);
  O2_SIGNPOST_START(data_processor_context, dpid, "callbacks", "Starting DataProcessorContext preExitCallbacks");
  // FIXME: we need to call the callback only once for the global services
  /// I guess...
  for (auto handle = handles.rbegin(); handle != handles.rend(); ++handle) {
    O2_SIGNPOST_ID_FROM_POINTER(cid, data_processor_context, handle->service);
    O2_SIGNPOST_START(data_processor_context, cid, "callbacks", "Starting DataProcessorContext::preExitCallbacks for service %{public}s", handle->spec.name.c_str());
    handle->callback(ref, handle->service);
    O2_SIGNPOST_END(data_processor_context, cid, "callbacks", "Ending DataProcessorContext::preExitCallbacks for service %{public}s", handle->spec.name.c_str());
  }
  O2_SIGNPOST_END(data_processor_context, dpid, "callbacks", "Ending DataProcessorContext preExitCallbacks");
}

/// Invoke callback to be executed on exit, in reverse order.
void DataProcessorContext::preLoopCallbacks(ServiceRegistryRef ref)
{
  invokeAll(preLoopHandles, "preLoopCallbacks", spec, ref);
}

void DataProcessorContext::domainInfoUpdatedCallback(ServiceRegistryRef ref, size_t oldestPossibleTimeslice, ChannelIndex channelIndex)
{
  O2_SIGNPOST_ID_FROM_POINTER(dpid, data_processor_context, this);
  O2_SIGNPOST_START(data_processor_context, dpid, "callbacks", "Starting DataProcessorContext domainInfoUpdatedCallback");
  for (auto& handle : domainInfoHandles) {
    O2_SIGNPOST_ID_FROM_POINTER(cid, data_processor_context, handle.service);
    O2_SIGNPOST_START(data_processor_context, cid, "callbacks", "Starting DataProcessorContext::domainInfoUpdatedCallback for service %{public}s", handle.spec.name.c_str());
    handle.callback(ref, oldestPossibleTimeslice, channelIndex);
    O2_SIGNPOST_END(data_processor_context, cid, "callbacks", "Ending DataProcessorContext::domainInfoUpdatedCallback for service %{public}s", handle.spec.name.c_str());
  }
  O2_SIGNPOST_END(data_processor_context, dpid, "callbacks", "Ending DataProcessorContext domainInfoUpdatedCallback");
}

void DataProcessorContext::preSendingMessagesCallbacks(ServiceRegistryRef ref, fair::mq::Parts& parts, ChannelIndex channelIndex)
{
  O2_SIGNPOST_ID_FROM_POINTER(dpid, data_processor_context, this);
  O2_SIGNPOST_START(data_processor_context, dpid, "callbacks", "Starting DataProcessorContext preSendingMessagesCallbacks");
  for (auto& handle : preSendingMessagesHandles) {
    O2_SIGNPOST_ID_FROM_POINTER(cid, data_processor_context, handle.service);
    O2_SIGNPOST_START(data_processor_context, cid, "callbacks", "Starting DataProcessorContext::preSendingMessagesCallbacks for service %{public}s", handle.spec.name.c_str());
    handle.callback(ref, parts, channelIndex);
    O2_SIGNPOST_END(data_processor_context, cid, "callbacks", "Ending DataProcessorContext::preSendingMessagesCallbacks for service %{public}s", handle.spec.name.c_str());
  }
  O2_SIGNPOST_END(data_processor_context, dpid, "callbacks", "Ending DataProcessorContext preSendingMessagesCallbacks");
}

} // namespace o2::framework
