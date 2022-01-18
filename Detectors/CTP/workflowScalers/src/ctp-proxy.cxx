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
#include <fairmq/FairMQDevice.h>
#include "CommonUtils/StringUtils.h"
#include <vector>
#include <string>

using namespace o2::framework;
using DetID = o2::detectors::DetID;

std::array<o2::header::DataOrigin, 1> exceptionsDetID{"GRP"};

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

InjectorFunction dcs2dpl()
{

  auto timesliceId = std::make_shared<size_t>(0);

  return [timesliceId](FairMQDevice& device, FairMQParts& parts, ChannelRetriever channelRetriever) {
    // make sure just 2 messages received
    if (parts.Size() != 2) {
      LOG(error) << "received " << parts.Size() << " instead of 2 expected";
      return;
    }
    std::string filename{static_cast<const char*>(parts.At(0)->GetData()), parts.At(0)->GetSize()};
    size_t filesize = parts.At(1)->GetSize();
    std::string filedata{static_cast<const char*>(parts.At(1)->GetData()), parts.At(1)->GetSize()};

    LOG(info) << "received file " << filename << " of size " << filesize << " Payload:" << filedata;
    o2::header::DataOrigin dataOrigin = getDataOriginFromFilename(filename);
    if (dataOrigin == o2::header::gDataOriginInvalid) {
      LOG(error) << "unknown detector for " << filename;
      return;
    }
    // o2::header::DataHeader hdrF("DCS_CONFIG_FILE", dataOrigin, 0);
    o2::header::DataHeader hdrN("CTP_COUNTERS", dataOrigin, 0);
    OutputSpec outsp{hdrN.dataOrigin, hdrN.dataDescription, hdrN.subSpecification};
    auto channel = channelRetriever(outsp, *timesliceId);
    if (channel.empty()) {
      LOG(error) << "No output channel found for OutputSpec " << outsp;
      return;
    }

    hdrN.tfCounter = *timesliceId; // this also
    hdrN.payloadSerializationMethod = o2::header::gSerializationMethodNone;
    hdrN.splitPayloadParts = 2;
    hdrN.splitPayloadIndex = 0;
    hdrN.payloadSize = parts.At(0)->GetSize();
    hdrN.firstTForbit = 0; // this should be irrelevant for DCS

    auto fmqFactory = device.GetChannel(channel).Transport();

    o2::header::Stack headerStackN{hdrN, DataProcessingHeader{*timesliceId, 0}};
    auto hdMessageN = fmqFactory->CreateMessage(headerStackN.size(), fair::mq::Alignment{64});
    auto plMessageN = fmqFactory->CreateMessage(hdrN.payloadSize, fair::mq::Alignment{64});
    memcpy(hdMessageN->GetData(), headerStackN.data(), headerStackN.size());
    memcpy(plMessageN->GetData(), parts.At(0)->GetData(), hdrN.payloadSize);

    FairMQParts outParts;
    outParts.AddPart(std::move(hdMessageN));
    outParts.AddPart(std::move(plMessageN));
    sendOnChannel(device, outParts, channel);

    LOG(info) << "Sent DPL message and acknowledgment for file " << filename;
  };
}

// we need to add workflow options before including Framework/runDataProcessing
void customize(std::vector<ConfigParamSpec>& workflowOptions)
{
  // workflowOptions.push_back(ConfigParamSpec{"subscribe-to", VariantType::String, "type=sub,method=connect,address=tcp://127.0.0.1:5556,rateLogging=1,transport=zeromq", {"channel subscribe to"}});
  workflowOptions.push_back(ConfigParamSpec{"subscribe-to", VariantType::String, "type=sub,method=connect,address=tcp://188.184.30.57:5556,rateLogging=1,transport=zeromq", {"channel subscribe to"}});
}

#include "Framework/runDataProcessing.h"

WorkflowSpec defineDataProcessing(ConfigContext const& config)
{
  LOG(info) << "Defining data processing";
  auto setChanName = [](const std::string& chan, const std::string& name) {
    size_t n = 0;
    if (std::string(chan).find("name=") != std::string::npos) {
      n = std::string(chan).find(",");
      if (n == std::string::npos) {
        throw std::runtime_error(fmt::format("wrongly formatted channel: {}", chan));
      }
      n++;
    }
    LOG(info) << "===>inside:" << name << " " << chan;
    return o2::utils::Str::concat_string("name=", name, ",", chan.substr(n, chan.size()));
  };
  const std::string devName = "ctp-proxy";
  auto chan = config.options().get<std::string>("subscribe-to");
  if (chan.empty()) {
    throw std::runtime_error("input channel is not provided");
  }
  chan = setChanName(chan, devName);
  LOG(info) << "name:" << devName << " chan:" << chan;
  LOG(info) << "Channels setup: " << chan;
  Outputs ctpCountersOutputs;
  ctpCountersOutputs.emplace_back("CTP", "CTP_COUNTERS", 0, Lifetime::Timeframe);
  LOG(info) << "===> Proxy to be set";
  DataProcessorSpec ctpProxy = specifyExternalFairMQDeviceProxy(
    devName.c_str(),
    std::move(ctpCountersOutputs),
    // this is just default, can be overriden by --dcs-config-proxy '--channel-config..'
    chan.c_str(),
    dcs2dpl());
  LOG(info) << "===> Proxy done";
  WorkflowSpec workflow;
  workflow.emplace_back(ctpProxy);
  return workflow;
}
