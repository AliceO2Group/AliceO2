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
#include "Framework/DataProcessingHelpers.h"
#include "Framework/SourceInfoHeader.h"
#include "Framework/DomainInfoHeader.h"
#include "Framework/ChannelSpec.h"
#include "Framework/ChannelInfo.h"
#include "MemoryResources/MemoryResources.h"
#include "Framework/FairMQDeviceProxy.h"
#include "Headers/DataHeader.h"
#include "Headers/DataHeaderHelpers.h"
#include "Headers/Stack.h"
#include "Framework/Logger.h"
#include "Framework/SendingPolicy.h"
#include "Framework/RawDeviceService.h"
#include "Framework/DeviceState.h"
#include "Framework/DeviceContext.h"
#include "Framework/ProcessingPolicies.h"
#include "Framework/Signpost.h"
#include "Framework/CallbackService.h"
#include "Framework/DefaultsHelpers.h"
#include "Framework/ServiceRegistryRef.h"
#include "Framework/DeviceSpec.h"
#include "Framework/ControlService.h"
#include "Framework/DataProcessingContext.h"
#include "Framework/DeviceStateEnums.h"
#include "Headers/DataHeader.h"
#include "Framework/DataProcessingHeader.h"
#include "DecongestionService.h"

#include <fairmq/Device.h>
#include <fairmq/Channel.h>

#include <uv.h>

// A log to use for general device logging
O2_DECLARE_DYNAMIC_LOG(device);
// Stream which keeps track of the calibration lifetime logic
O2_DECLARE_DYNAMIC_LOG(calibration);
O2_DECLARE_DYNAMIC_LOG(forwarding);

namespace o2::framework
{
void DataProcessingHelpers::sendEndOfStream(ServiceRegistryRef const& ref, OutputChannelSpec const& channel)
{
  fair::mq::Device* device = ref.get<RawDeviceService>().device();
  fair::mq::Parts parts;
  fair::mq::MessagePtr payload(device->NewMessage());
  SourceInfoHeader sih;
  sih.state = InputChannelState::Completed;
  auto channelAlloc = o2::pmr::getTransportAllocator(device->GetChannel(channel.name, 0).Transport());
  auto header = o2::pmr::getMessage(o2::header::Stack{channelAlloc, sih});
  // sigh... See if we can avoid having it const by not
  // exposing it to the user in the first place.
  parts.AddPart(std::move(header));
  parts.AddPart(std::move(payload));
  device->Send(parts, channel.name, 0);
  LOGP(info, "Sending end-of-stream message to channel {}", channel.name);
}

void doSendOldestPossibleTimeframe(ServiceRegistryRef ref, fair::mq::TransportFactory* transport, ChannelIndex index, SendingPolicy::SendingCallback const& callback, size_t timeslice)
{
  fair::mq::Parts parts;
  fair::mq::MessagePtr payload(transport->CreateMessage());
  o2::framework::DomainInfoHeader dih;
  dih.oldestPossibleTimeslice = timeslice;
  auto channelAlloc = o2::pmr::getTransportAllocator(transport);
  auto header = o2::pmr::getMessage(o2::header::Stack{channelAlloc, dih});
  // sigh... See if we can avoid having it const by not
  // exposing it to the user in the first place.
  parts.AddPart(std::move(header));
  parts.AddPart(std::move(payload));

  callback(parts, index, ref);
}

bool DataProcessingHelpers::sendOldestPossibleTimeframe(ServiceRegistryRef const& ref, ForwardChannelInfo const& info, ForwardChannelState& state, size_t timeslice)
{
  if (ref.get<DecongestionService>().suppressDomainInfo) {
    return false;
  }
  if (state.oldestForChannel.value >= timeslice) {
    return false;
  }
  doSendOldestPossibleTimeframe(ref, info.channel.Transport(), info.index, info.policy->forward, timeslice);
  state.oldestForChannel = {timeslice};
  return true;
}

bool DataProcessingHelpers::sendOldestPossibleTimeframe(ServiceRegistryRef const& ref, OutputChannelInfo const& info, OutputChannelState& state, size_t timeslice)
{
  if (ref.get<DecongestionService>().suppressDomainInfo) {
    return false;
  }
  if (state.oldestForChannel.value >= timeslice) {
    return false;
  }
  doSendOldestPossibleTimeframe(ref, info.channel.Transport(), info.index, info.policy->send, timeslice);
  state.oldestForChannel = {timeslice};
  return true;
}

void DataProcessingHelpers::broadcastOldestPossibleTimeslice(ServiceRegistryRef const& ref, size_t timeslice)
{
  auto& proxy = ref.get<FairMQDeviceProxy>();
  for (int ci = 0; ci < proxy.getNumOutputChannels(); ++ci) {
    auto& info = proxy.getOutputChannelInfo({ci});
    auto& state = proxy.getOutputChannelState({ci});
    sendOldestPossibleTimeframe(ref, info, state, timeslice);
  }
}

void DataProcessingHelpers::switchState(ServiceRegistryRef const& ref, StreamingState newState)
{
  auto& state = ref.get<DeviceState>();
  auto& context = ref.get<DataProcessorContext>();
  O2_SIGNPOST_ID_FROM_POINTER(dpid, device, &context);
  O2_SIGNPOST_END(device, dpid, "state", "End of processing state %d", (int)state.streaming);
  O2_SIGNPOST_START(device, dpid, "state", "Starting processing state %d", (int)newState);
  state.streaming = newState;
  ref.get<ControlService>().notifyStreamingState(state.streaming);
};

bool hasOnlyTimers(DeviceSpec const& spec)
{
  return std::all_of(spec.inputs.cbegin(), spec.inputs.cend(), [](InputRoute const& route) -> bool { return route.matcher.lifetime == Lifetime::Timer; });
}

bool DataProcessingHelpers::hasOnlyGenerated(DeviceSpec const& spec)
{
  return (spec.inputChannels.size() == 1) && (spec.inputs[0].matcher.lifetime == Lifetime::Timer || spec.inputs[0].matcher.lifetime == Lifetime::Enumeration);
}

void on_data_processing_expired(uv_timer_t* handle)
{
  auto* ref = (ServiceRegistryRef*)handle->data;
  auto& state = ref->get<DeviceState>();
  auto& spec = ref->get<DeviceSpec const>();
  state.loopReason |= DeviceState::TIMER_EXPIRED;

  // Check if this is a source device
  O2_SIGNPOST_ID_FROM_POINTER(cid, calibration, handle);

  if (DataProcessingHelpers::hasOnlyGenerated(spec)) {
    O2_SIGNPOST_EVENT_EMIT_INFO(calibration, cid, "callback", "Grace period for data processing expired. Switching to EndOfStreaming.");
    DataProcessingHelpers::switchState(*ref, StreamingState::EndOfStreaming);
  } else {
    O2_SIGNPOST_EVENT_EMIT_INFO(calibration, cid, "callback", "Grace period for data processing expired. Only calibrations from this point onwards.");
    state.allowedProcessing = DeviceState::CalibrationOnly;
  }
}

void on_transition_requested_expired(uv_timer_t* handle)
{
  auto* ref = (ServiceRegistryRef*)handle->data;
  auto& state = ref->get<DeviceState>();
  state.loopReason |= DeviceState::TIMER_EXPIRED;
  // Check if this is a source device
  O2_SIGNPOST_ID_FROM_POINTER(cid, calibration, handle);
  auto& spec = ref->get<DeviceSpec const>();
  std::string messageOnExpire = DataProcessingHelpers::hasOnlyGenerated(spec) ? "DPL exit transition grace period for source expired. Exiting." : fmt::format("DPL exit transition grace period for {} expired. Exiting.", state.allowedProcessing == DeviceState::CalibrationOnly ? "calibration" : "data & calibration").c_str();
  if (!ref->get<RawDeviceService>().device()->GetConfig()->GetValue<bool>("error-on-exit-transition-timeout")) {
    O2_SIGNPOST_EVENT_EMIT_WARN(calibration, cid, "callback", "%{public}s", messageOnExpire.c_str());
  } else {
    O2_SIGNPOST_EVENT_EMIT_ERROR(calibration, cid, "callback", "%{public}s", messageOnExpire.c_str());
  }
  state.transitionHandling = TransitionHandlingState::Expired;
}

TransitionHandlingState DataProcessingHelpers::updateStateTransition(ServiceRegistryRef const& ref, ProcessingPolicies const& policies)
{
  auto& state = ref.get<DeviceState>();
  auto& deviceProxy = ref.get<FairMQDeviceProxy>();
  if (state.transitionHandling != TransitionHandlingState::NoTransition || deviceProxy.newStateRequested() == false) {
    return state.transitionHandling;
  }
  O2_SIGNPOST_ID_FROM_POINTER(lid, device, state.loop);
  O2_SIGNPOST_ID_FROM_POINTER(cid, calibration, state.loop);
  auto& deviceContext = ref.get<DeviceContext>();
  // Check if we only have timers
  auto& spec = ref.get<DeviceSpec const>();
  if (hasOnlyTimers(spec)) {
    DataProcessingHelpers::switchState(ref, StreamingState::EndOfStreaming);
  }

  // We do not do anything in particular if the data processing timeout would go past the exitTransitionTimeout
  if (deviceContext.dataProcessingTimeout > 0 && deviceContext.dataProcessingTimeout < deviceContext.exitTransitionTimeout) {
    uv_update_time(state.loop);
    O2_SIGNPOST_EVENT_EMIT(calibration, cid, "timer_setup", "Starting %d s timer for dataProcessingTimeout.", deviceContext.dataProcessingTimeout);
    uv_timer_start(deviceContext.dataProcessingGracePeriodTimer, on_data_processing_expired, deviceContext.dataProcessingTimeout * 1000, 0);
  }
  if (deviceContext.exitTransitionTimeout != 0 && state.streaming != StreamingState::Idle) {
    ref.get<CallbackService>().call<CallbackService::Id::ExitRequested>(ServiceRegistryRef{ref});
    uv_update_time(state.loop);
    O2_SIGNPOST_EVENT_EMIT(calibration, cid, "timer_setup", "Starting %d s timer for exitTransitionTimeout.",
                           deviceContext.exitTransitionTimeout);
    uv_timer_start(deviceContext.gracePeriodTimer, on_transition_requested_expired, deviceContext.exitTransitionTimeout * 1000, 0);
    bool onlyGenerated = DataProcessingHelpers::hasOnlyGenerated(spec);
    int timeout = onlyGenerated ? deviceContext.dataProcessingTimeout : deviceContext.exitTransitionTimeout;
    if (policies.termination == TerminationPolicy::QUIT && DefaultsHelpers::onlineDeploymentMode() == false) {
      O2_SIGNPOST_EVENT_EMIT_INFO(device, lid, "run_loop", "New state requested. Waiting for %d seconds before quitting.", timeout);
    } else {
      O2_SIGNPOST_EVENT_EMIT_INFO(device, lid, "run_loop",
                                  "New state requested. Waiting for %d seconds before %{public}s",
                                  timeout,
                                  onlyGenerated ? "dropping remaining input and switching to READY state." : "switching to READY state.");
    }
    return TransitionHandlingState::Requested;
  } else {
    if (deviceContext.exitTransitionTimeout == 0 && policies.termination == TerminationPolicy::QUIT) {
      O2_SIGNPOST_EVENT_EMIT_INFO(device, lid, "run_loop", "New state requested. No timeout set, quitting immediately as per --completion-policy");
    } else if (deviceContext.exitTransitionTimeout == 0 && policies.termination != TerminationPolicy::QUIT) {
      O2_SIGNPOST_EVENT_EMIT_INFO(device, lid, "run_loop", "New state requested. No timeout set, switching to READY state immediately");
    } else if (policies.termination == TerminationPolicy::QUIT) {
      O2_SIGNPOST_EVENT_EMIT_INFO(device, lid, "run_loop", "New state pending and we are already idle, quitting immediately as per --completion-policy");
    } else {
      O2_SIGNPOST_EVENT_EMIT_INFO(device, lid, "run_loop", "New state pending and we are already idle, switching to READY immediately.");
    }
    return TransitionHandlingState::Expired;
  }
}

void DataProcessingHelpers::routeForwardedMessages(FairMQDeviceProxy& proxy, std::span<fair::mq::MessagePtr>& messages, std::vector<fair::mq::Parts>& forwardedParts,
                                                   const bool copyByDefault, bool consume)
{
  O2_SIGNPOST_ID_GENERATE(sid, forwarding);
  std::vector<ChannelIndex> forwardingChoices{};
  size_t pi = 0;
  while (pi < messages.size()) {
    auto& header = messages[pi];

    // If is now possible that the record is not complete when
    // we forward it, because of a custom completion policy.
    // this means that we need to skip the empty entries in the
    // record for being forwarded.
    if (header->GetData() == nullptr) {
      pi += 2;
      continue;
    }
    auto dih = o2::header::get<DomainInfoHeader*>(header->GetData());
    if (dih) {
      pi += 2;
      continue;
    }
    auto sih = o2::header::get<SourceInfoHeader*>(header->GetData());
    if (sih) {
      pi += 2;
      continue;
    }

    auto dph = o2::header::get<DataProcessingHeader*>(header->GetData());
    auto dh = o2::header::get<o2::header::DataHeader*>(header->GetData());

    if (dph == nullptr || dh == nullptr) {
      // Complain only if this is not an out-of-band message
      LOGP(error, "Data is missing {}{}{}",
           dph ? "DataProcessingHeader" : "", dph || dh ? "and" : "", dh ? "DataHeader" : "");
      pi += 2;
      continue;
    }

    // At least one payload.
    auto& payload = messages[pi + 1];
    // Calculate the number of messages which should be handled together
    // all in one go.
    size_t numberOfMessages = 0;
    if (dh->splitPayloadParts > 0 && dh->splitPayloadParts == dh->splitPayloadIndex) {
      // Sequence of (header, payload[0], ... , payload[splitPayloadParts - 1]) pairs belonging together.
      numberOfMessages = dh->splitPayloadParts + 1; // one is for the header
    } else {
      // Sequence of splitPayloadParts (header, payload) pairs belonging together.
      // In case splitPayloadParts = 0, we consider this as a single message pair
      numberOfMessages = (dh->splitPayloadParts > 0 ? dh->splitPayloadParts : 1) * 2;
    }

    if (payload.get() == nullptr && consume == true) {
      // If the payload is not there, it means we already
      // processed it with ConsumeExisiting. Therefore we
      // need to do something only if this is the last consume.
      header.reset(nullptr);
      pi += numberOfMessages;
      continue;
    }

    // We need to find the forward route only for the first
    // part of a split payload. All the others will use the same.
    // Therefore, we reset and recompute the forwarding choice:
    //
    // - If this is the first payload of a [header0][payload0][header0][payload1]... sequence,
    //   which is actually always created and handled together. Notice that in this
    //   case we have splitPayloadParts == splitPayloadIndex
    // - If this is the first payload of a [header0][payload0][header1][payload1]... sequence
    //   belonging to the same multipart message (and therefore we are guaranteed that they
    //   need to be routed together).
    // - If the message is not a multipart (splitPayloadParts 0) or has only one part
    // - If it's a message of the kind [header0][payload1][payload2][payload3]... and therefore
    //   we will already use the same choice in the for loop below.
    //

    forwardingChoices.clear();
    proxy.getMatchingForwardChannelIndexes(forwardingChoices, *dh, dph->startTime);

    if (forwardingChoices.empty()) {
      // Nothing to forward go to the next messageset
      pi += numberOfMessages;
      continue;
    }

    // In case of more than one forward route, we need to copy the message.
    // This will eventually use the same memory if running with the same backend.
    if (copyByDefault || forwardingChoices.size() > 1) {
      for (auto& choice : forwardingChoices) {
        O2_SIGNPOST_EVENT_EMIT(forwarding, sid, "forwardInputs", "Forwarding a copy of %{public}s to route %d.",
                               fmt::format("{}/{}/{}@timeslice:{} tfCounter:{}", dh->dataOrigin, dh->dataDescription, dh->subSpecification, dph->startTime, dh->tfCounter).c_str(), choice.value);

        for (size_t ppi = pi; ppi < pi + numberOfMessages; ++ppi) {
          auto&& newMsg = header->GetTransport()->CreateMessage();
          newMsg->Copy(*messages[ppi]);
          forwardedParts[choice.value].AddPart(std::move(newMsg));
        }
      }
    } else {
      O2_SIGNPOST_EVENT_EMIT(forwarding, sid, "forwardInputs", "Forwarding %{public}s to route %d.",
                             fmt::format("{}/{}/{}@timeslice:{} tfCounter:{}", dh->dataOrigin, dh->dataDescription, dh->subSpecification, dph->startTime, dh->tfCounter).c_str(), forwardingChoices.back().value);
      for (size_t ppi = pi; ppi < pi + numberOfMessages; ++ppi) {
        forwardedParts[forwardingChoices.back().value].AddPart(std::move(messages[ppi]));
      }
    }
    pi += numberOfMessages;
  }
}

auto DataProcessingHelpers::routeForwardedMessageSet(FairMQDeviceProxy& proxy,
                                                     std::vector<MessageSet>& currentSetOfInputs,
                                                     const bool copyByDefault, bool consume) -> std::vector<fair::mq::Parts>
{
  // we collect all messages per forward in a map and send them together
  std::vector<fair::mq::Parts> forwardedParts(proxy.getNumForwardChannels());

  for (size_t ii = 0, ie = currentSetOfInputs.size(); ii < ie; ++ii) {
    auto span = std::span<fair::mq::MessagePtr>(currentSetOfInputs[ii].messages);
    routeForwardedMessages(proxy, span, forwardedParts, copyByDefault, consume);
  }
  return forwardedParts;
};

} // namespace o2::framework
