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

// example to run:
// o2-dcs-config-proxy --dcs-config-proxy '--channel-config "name=dcs-config-proxy,type=sub,method=connect,address=tcp://127.0.0.1:5556,rateLogging=1,transport=zeromq"' \
//                     --acknowlege-to "type=push,method=connect,address=tcp://127.0.0.1:5557,rateLogging=1,transport=zeromq"

#include "Framework/WorkflowSpec.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/DataSpecUtils.h"
#include "Framework/ControlService.h"
#include "Framework/Logger.h"
#include "Framework/Lifetime.h"
#include "Framework/ConfigParamSpec.h"
#include "Framework/ExternalFairMQDeviceProxy.h"
#include "DetectorsCommonDataFormats/DetID.h"
#include "Headers/DataHeaderHelpers.h"
#include <fairmq/Device.h>
#include <fairmq/Parts.h>
#include "CommonUtils/StringUtils.h"
#include <vector>
#include <string>
#include <chrono>

using namespace o2::framework;
using DetID = o2::detectors::DetID;

std::array<o2::header::DataOrigin, 1> exceptionsDetID{"GRP"};

void sendAnswer(const std::string& what, const std::string& ack_chan, fair::mq::Device& device)
{
  if (!ack_chan.empty()) {
    auto fmqFactory = device.GetChannel(ack_chan).Transport();
    auto msg = fmqFactory->CreateMessage(what.size(), fair::mq::Alignment{64});
    memcpy(msg->GetData(), what.c_str(), what.size());
    fair::mq::Parts outParts;
    outParts.AddPart(std::move(msg));
    sendOnChannel(device, outParts, ack_chan, (size_t)-1);
  }
}

auto getDataOriginFromFilename(const std::string& filename)
{
  // assume the filename start with detector name
  auto dIDStr = filename.substr(0, 3);
  auto dID = DetID::nameToID(dIDStr.c_str(), DetID::First);
  o2::header::DataOrigin dataOrigin;
  if (dID < 0) {
    for (auto& el : exceptionsDetID) {
      if (el.as<std::string>() == dIDStr) {
        return el;
      }
    }
    return o2::header::gDataOriginInvalid;
  }
  return DetID(dID).getDataOrigin();
}

InjectorFunction dcs2dpl(const std::string& acknowledge)
{

  auto timesliceId = std::make_shared<size_t>(0);

  return [acknowledge, timesliceId](TimingInfo&, fair::mq::Device& device, fair::mq::Parts& parts, ChannelRetriever channelRetriever) {
    if (parts.Size() == 0) { // received at ^c, ignore
      LOG(info) << "ignoring empty message";
      return;
    }
    // make sure just 2 messages received
    if (parts.Size() != 2) {
      LOG(error) << "received " << parts.Size() << " instead of 2 expected";
      sendAnswer("error0: wrong number of messages", acknowledge, device);
      return;
    }
    std::string filename{static_cast<const char*>(parts.At(0)->GetData()), parts.At(0)->GetSize()};
    size_t filesize = parts.At(1)->GetSize();
    LOG(info) << "received file " << filename << " of size " << filesize;
    o2::header::DataOrigin dataOrigin = getDataOriginFromFilename(filename);
    if (dataOrigin == o2::header::gDataOriginInvalid) {
      LOG(error) << "unknown detector for " << filename;
      sendAnswer(fmt::format("{}:error1: unrecognized filename", filename), acknowledge, device);
      return;
    }

    o2::header::DataHeader hdrF("DCS_CONFIG_FILE", dataOrigin, 0);
    o2::header::DataHeader hdrN("DCS_CONFIG_NAME", dataOrigin, 0);
    OutputSpec outsp{hdrF.dataOrigin, hdrF.dataDescription, hdrF.subSpecification};
    auto channel = channelRetriever(outsp, *timesliceId);
    if (channel.empty()) {
      LOG(error) << "No output channel found for OutputSpec " << outsp;
      sendAnswer(fmt::format("{}:error2: no channel to send", filename), acknowledge, device);
      return;
    }

    hdrF.tfCounter = *timesliceId;
    hdrF.payloadSerializationMethod = o2::header::gSerializationMethodNone;
    hdrF.splitPayloadParts = 1;
    hdrF.splitPayloadIndex = 0;
    hdrF.payloadSize = filesize;
    hdrF.firstTForbit = 0; // this should be irrelevant for DCS

    hdrN.tfCounter = *timesliceId;
    hdrN.payloadSerializationMethod = o2::header::gSerializationMethodNone;
    hdrN.splitPayloadParts = 1;
    hdrN.splitPayloadIndex = 0;
    hdrN.payloadSize = parts.At(0)->GetSize();
    hdrN.firstTForbit = 0; // this should be irrelevant for DCS

    auto fmqFactory = device.GetChannel(channel).Transport();
    std::uint64_t creation = std::chrono::time_point_cast<std::chrono::milliseconds>(std::chrono::system_clock::now()).time_since_epoch().count();

    o2::header::Stack headerStackF{hdrF, DataProcessingHeader{*timesliceId, 1, creation}};
    auto hdMessageF = fmqFactory->CreateMessage(headerStackF.size(), fair::mq::Alignment{64});
    auto plMessageF = fmqFactory->CreateMessage(hdrF.payloadSize, fair::mq::Alignment{64});
    memcpy(hdMessageF->GetData(), headerStackF.data(), headerStackF.size());
    memcpy(plMessageF->GetData(), parts.At(1)->GetData(), hdrF.payloadSize);

    o2::header::Stack headerStackN{hdrN, DataProcessingHeader{*timesliceId, 1, creation}};
    auto hdMessageN = fmqFactory->CreateMessage(headerStackN.size(), fair::mq::Alignment{64});
    auto plMessageN = fmqFactory->CreateMessage(hdrN.payloadSize, fair::mq::Alignment{64});
    memcpy(hdMessageN->GetData(), headerStackN.data(), headerStackN.size());
    memcpy(plMessageN->GetData(), parts.At(0)->GetData(), hdrN.payloadSize);

    fair::mq::Parts outPartsF;
    outPartsF.AddPart(std::move(hdMessageF));
    outPartsF.AddPart(std::move(plMessageF));
    sendOnChannel(device, outPartsF, channel, (size_t)-1);

    fair::mq::Parts outPartsN;
    outPartsN.AddPart(std::move(hdMessageN));
    outPartsN.AddPart(std::move(plMessageN));
    sendOnChannel(device, outPartsN, channel, *timesliceId);

    sendAnswer(fmt::format("{}:ok", filename), acknowledge, device);
    LOG(info) << "Sent DPL message and acknowledgment for file " << filename;
    (*timesliceId)++;
  };
}

// we need to add workflow options before including Framework/runDataProcessing
void customize(std::vector<ConfigParamSpec>& workflowOptions)
{
  workflowOptions.push_back(ConfigParamSpec{"acknowlege-to", VariantType::String, "type=push,method=connect,address=tcp://127.0.0.1:5557,rateLogging=1,transport=zeromq", {"channel to acknowledge, no acknowledgement if empty"}});
  workflowOptions.push_back(ConfigParamSpec{"subscribe-to", VariantType::String, "type=sub,method=connect,address=tcp://127.0.0.1:5556,rateLogging=1,transport=zeromq", {"channel subscribe to"}});
}

#include "Framework/runDataProcessing.h"

WorkflowSpec defineDataProcessing(ConfigContext const& config)
{
  auto setChanName = [](const std::string& chan, const std::string& name) {
    size_t n = 0;
    if (std::string(chan).find("name=") != std::string::npos) {
      n = std::string(chan).find(",");
      if (n == std::string::npos) {
        throw std::runtime_error(fmt::format("wrongly formatted channel: {}", chan));
      }
      n++;
    }
    return o2::utils::Str::concat_string("name=", name, ",", chan.substr(n, chan.size()));
  };

  const std::string devName = "dcs-config-proxy";
  auto chan = config.options().get<std::string>("subscribe-to");
  if (chan.empty()) {
    throw std::runtime_error("input channel is not provided");
  }
  chan = setChanName(chan, devName);

  auto chanTo = config.options().get<std::string>("acknowlege-to");
  std::string ackChan{};
  if (!chanTo.empty()) {
    ackChan = "ackChan";
    chan = o2::utils::Str::concat_string(chan, ";", setChanName(chanTo, ackChan));
  }
  LOG(info) << "Channels setup: " << chan;
  Outputs dcsOutputs;

  for (int id = DetID::First; id <= DetID::Last; id++) {
    dcsOutputs.emplace_back(DetID(id).getDataOrigin(), "DCS_CONFIG_FILE", 0, Lifetime::Sporadic);
    dcsOutputs.emplace_back(DetID(id).getDataOrigin(), "DCS_CONFIG_NAME", 0, Lifetime::Sporadic);
  }
  for (auto& el : exceptionsDetID) {
    dcsOutputs.emplace_back(el, "DCS_CONFIG_FILE", 0, Lifetime::Sporadic);
    dcsOutputs.emplace_back(el, "DCS_CONFIG_NAME", 0, Lifetime::Sporadic);
  }

  DataProcessorSpec dcsConfigProxy = specifyExternalFairMQDeviceProxy(
    devName.c_str(),
    std::move(dcsOutputs),
    // this is just default, can be overriden by --dcs-config-proxy '--channel-config..'
    chan.c_str(),
    dcs2dpl(ackChan));

  WorkflowSpec workflow;
  workflow.emplace_back(dcsConfigProxy);
  return workflow;
}
