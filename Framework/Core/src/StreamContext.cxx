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

#include "Framework/StreamContext.h"

#include "Framework/Signpost.h"

O2_DECLARE_DYNAMIC_LOG(stream_context);

namespace o2::framework
{

void StreamContext::preStartStreamCallbacks(ServiceRegistryRef ref)
{
  for (auto& handle : preStartStreamHandles) {
    LOG(detail) << "Invoking preStartStreamCallbacks for " << handle.spec.name;
    assert(handle.callback);
    // The service must be nullptr because we have not created it yet at this
    // point.
    handle.callback(ref, nullptr);
  }
}
/// Invoke callbacks to be executed before every process method invokation
void StreamContext::preProcessingCallbacks(ProcessingContext& pcx)
{
  for (auto& handle : preProcessingHandles) {
    LOG(debug) << "Invoking preProcessingCallbacks for" << handle.service;
    assert(handle.service);
    assert(handle.callback);
    handle.callback(pcx, handle.service);
  }
}

/// Invoke callbacks to be executed after every process method invokation
void StreamContext::finaliseOutputsCallbacks(ProcessingContext& pcx)
{
  for (auto& handle : finaliseOutputsHandles) {
    LOG(debug) << "Invoking finaliseOutputsCallbacks for " << handle.service;
    assert(handle.service);
    assert(handle.callback);
    handle.callback(pcx, handle.service);
  }
}

/// Invoke callbacks to be executed after every process method invokation
void StreamContext::postProcessingCallbacks(ProcessingContext& pcx)
{
  O2_SIGNPOST_ID_FROM_POINTER(dpid, stream_context, &pcx);
  O2_SIGNPOST_START(stream_context, dpid, "callbacks", "Starting StreamContext postProcessingCallbacks");
  for (auto& handle : postProcessingHandles) {
    O2_SIGNPOST_ID_FROM_POINTER(cid, stream_context, handle.service);
    O2_SIGNPOST_START(stream_context, cid, "callbacks", "Starting StreamContext::postProcessingCallbacks for service %{public}s", handle.spec.name.c_str());
    assert(handle.service);
    assert(handle.callback);
    handle.callback(pcx, handle.service);
    O2_SIGNPOST_END(stream_context, cid, "callbacks", "Ending StreamContext::postProcessingCallbacks for service %{public}s", handle.spec.name.c_str());
  }
  O2_SIGNPOST_END(stream_context, dpid, "callbacks", "Ending StreamContext postProcessingCallbacks");
}

/// Invoke callbacks to be executed before every EOS user callback invokation
void StreamContext::preEOSCallbacks(EndOfStreamContext& eosContext)
{
  for (auto& eosHandle : preEOSHandles) {
    LOG(detail) << "Invoking preEOSCallbacks for" << eosHandle.service;
    assert(eosHandle.service);
    assert(eosHandle.callback);
    eosHandle.callback(eosContext, eosHandle.service);
  }
}

/// Invoke callbacks to be executed after every EOS user callback invokation
void StreamContext::postEOSCallbacks(EndOfStreamContext& eosContext)
{
  for (auto& eosHandle : postEOSHandles) {
    LOG(detail) << "Invoking postEOSCallbacks for " << eosHandle.service;
    assert(eosHandle.service);
    assert(eosHandle.callback);
    eosHandle.callback(eosContext, eosHandle.service);
  }
}

} // namespace o2::framework
