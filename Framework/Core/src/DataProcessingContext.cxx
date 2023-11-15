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

namespace o2::framework
{
/// Invoke callbacks to be executed before every dangling check
void DataProcessorContext::preProcessingCallbacks(ProcessingContext& ctx)
{
  for (auto& handle : preProcessingHandlers) {
    LOGP(debug, "Invoking preDanglingCallback for service {}", handle.spec.name);
    handle.callback(ctx, handle.service);
  }
}

void DataProcessorContext::finaliseOutputsCallbacks(ProcessingContext& ctx)
{
  for (auto& handle : finaliseOutputsHandles) {
    LOGP(debug, "Invoking postProcessingCallback for service {}", handle.spec.name);
    handle.callback(ctx, handle.service);
  }
}

/// Invoke callbacks to be executed before every dangling check
void DataProcessorContext::postProcessingCallbacks(ProcessingContext& ctx)
{
  for (auto& handle : postProcessingHandlers) {
    LOGP(debug, "Invoking postProcessingCallback for service {}", handle.spec.name);
    handle.callback(ctx, handle.service);
  }
}

/// Invoke callbacks to be executed before every dangling check
void DataProcessorContext::preDanglingCallbacks(DanglingContext& danglingContext)
{
  for (auto& handle : preDanglingHandles) {
    LOGP(debug, "Invoking preDanglingCallback for service {}", handle.spec.name);
    handle.callback(danglingContext, handle.service);
  }
}

/// Invoke callbacks to be executed after every dangling check
void DataProcessorContext::postDanglingCallbacks(DanglingContext& danglingContext)
{
  for (auto& handle : postDanglingHandles) {
    LOGP(debug, "Invoking postDanglingCallback for service {}", handle.spec.name);
    handle.callback(danglingContext, handle.service);
  }
}

/// Invoke callbacks to be executed before every EOS user callback invokation
void DataProcessorContext::preEOSCallbacks(EndOfStreamContext& eosContext)
{
  for (auto& handle : preEOSHandles) {
    LOGP(detail, "Invoking preEosCallback for service {}", handle.spec.name);
    handle.callback(eosContext, handle.service);
  }
}

/// Invoke callbacks to be executed after every EOS user callback invokation
void DataProcessorContext::postEOSCallbacks(EndOfStreamContext& eosContext)
{
  for (auto& handle : postEOSHandles) {
    LOGP(detail, "Invoking postEoSCallback for service {}", handle.spec.name);
    handle.callback(eosContext, handle.service);
  }
}

/// Invoke callbacks to be executed after every data Dispatching
void DataProcessorContext::postDispatchingCallbacks(ProcessingContext& processContext)
{
  for (auto& handle : postDispatchingHandles) {
    LOGP(debug, "Invoking postDispatchingCallback for service {}", handle.spec.name);
    handle.callback(processContext, handle.service);
  }
}

/// Invoke callbacks to be executed after every data Dispatching
void DataProcessorContext::postForwardingCallbacks(ProcessingContext& processContext)
{
  for (auto& handle : postForwardingHandles) {
    LOGP(debug, "Invoking postForwardingCallback for service {}", handle.spec.name);
    handle.callback(processContext, handle.service);
  }
}

/// Callbacks to be called in fair::mq::Device::PreRun()
void DataProcessorContext::preStartCallbacks(ServiceRegistryRef ref)
{
  for (auto& handle : preStartHandles) {
    LOGP(detail, "Invoking preStartCallback for service {}", handle.spec.name);
    handle.callback(ref, handle.service);
  }
}

void DataProcessorContext::postStopCallbacks(ServiceRegistryRef ref)
{
  // FIXME: we need to call the callback only once for the global services
  /// I guess...
  for (auto& handle : postStopHandles) {
    LOGP(detail, "Invoking postStopCallback for service {}", handle.spec.name);
    handle.callback(ref, handle.service);
  }
}

/// Invoke callback to be executed on exit, in reverse order.
void DataProcessorContext::preExitCallbacks(std::vector<ServiceExitHandle> handles, ServiceRegistryRef ref)
{
  // FIXME: we need to call the callback only once for the global services
  /// I guess...
  for (auto handle = handles.rbegin(); handle != handles.rend(); ++handle) {
    LOGP(detail, "Invoking preExitCallback for service {}", handle->spec.name);
    handle->callback(ref, handle->service);
  }
}

/// Invoke callback to be executed on exit, in reverse order.
void DataProcessorContext::preLoopCallbacks(ServiceRegistryRef ref)
{
  // FIXME: we need to call the callback only once for the global services
  /// I guess...
  LOGP(debug, "Invoking preLoopCallbacks");
  for (auto& handle : preLoopHandles) {
    LOGP(debug, "Invoking preLoopCallback for service {}", handle.spec.name);
    handle.callback(ref, handle.service);
  }
}

void DataProcessorContext::domainInfoUpdatedCallback(ServiceRegistryRef ref, size_t oldestPossibleTimeslice, ChannelIndex channelIndex)
{
  for (auto& handle : domainInfoHandles) {
    LOGP(debug, "Invoking domainInfoHandles for service {}", handle.spec.name);
    handle.callback(ref, oldestPossibleTimeslice, channelIndex);
  }
}

void DataProcessorContext::preSendingMessagesCallbacks(ServiceRegistryRef ref, fair::mq::Parts& parts, ChannelIndex channelIndex)
{
  for (auto& handle : preSendingMessagesHandles) {
    LOGP(debug, "Invoking preSending for service {}", handle.spec.name);
    handle.callback(ref, parts, channelIndex);
  }
}

} // namespace o2::framework
