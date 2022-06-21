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
// default: processing , intermal database
// o2-ctp-proxy --ctp-proxy '--channel-config "name=ctp-proxy,type=sub,method=connect,address=tcp://10.161.64.100:50090,rateLogging=5,transport=zeromq"' -b
// no processing
// o2-ctp-proxy --ctp-proxy '--channel-config "name=ctp-proxy,type=sub,method=connect,address=tcp://10.161.64.100:50090,rateLogging=5,transport=zeromq"' --for-qc -b
// processing, test database
// o2-ctp-proxy --ctp-proxy '--channel-config "name=ctp-proxy,type=sub,method=connect,address=tcp://10.161.64.100:50090,rateLogging=5,transport=zeromq"' '--ccdb-host=http://ccdb-test.cern.ch:8080' -b

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
#include "DataFormatsCTP/Configuration.h"
#include <vector>
#include <string>

using namespace o2::framework;
using DetID = o2::detectors::DetID;
InjectorFunction dcs2dpl(bool qc, std::string& ccdbhost)
// InjectorFunction dcs2dpl()
{
  auto timesliceId = std::make_shared<size_t>(0);
  auto runMgr = std::make_shared<o2::ctp::CTPRunManager>();
  runMgr->setCCDBHost(ccdbhost);
  runMgr->setCTPQC(qc);
  runMgr->init();
  return [timesliceId, runMgr](TimingInfo&, fair::mq::Device& device, fair::mq::Parts& parts, ChannelRetriever channelRetriever) {
    // make sure just 2 messages received
    if (parts.Size() != 2) {
      LOG(error) << "received " << parts.Size() << " instead of 2 expected";
      return;
    }
    std::string messageHeader{static_cast<const char*>(parts.At(0)->GetData()), parts.At(0)->GetSize()};
    size_t dataSize = parts.At(1)->GetSize();
    std::string messageData{static_cast<const char*>(parts.At(1)->GetData()), parts.At(1)->GetSize()};
    LOG(info) << "received message " << messageHeader << " of size " << dataSize; // << " Payload:" << messageData;
    o2::header::DataHeader hdrF("CTP_COUNTERS", o2::header::gDataOriginCTP, 0);
    OutputSpec outsp{hdrF.dataOrigin, hdrF.dataDescription, hdrF.subSpecification};
    auto channel = channelRetriever(outsp, *timesliceId);
    if (channel.empty()) {
      LOG(error) << "No output channel found for OutputSpec " << outsp;
      return;
    }
    runMgr->processMessage(messageHeader, messageData);
    hdrF.tfCounter = *timesliceId; // this also
    hdrF.payloadSerializationMethod = o2::header::gSerializationMethodNone;
    hdrF.splitPayloadParts = 1;
    hdrF.splitPayloadIndex = 0;
    hdrF.payloadSize = parts.At(1)->GetSize();
    hdrF.firstTForbit = 0; // this should be irrelevant for Counters ? Orbit is in payload

    auto fmqFactory = device.GetChannel(channel).Transport();

    o2::header::Stack headerStackF{hdrF, DataProcessingHeader{*timesliceId, 1}};
    auto hdMessageF = fmqFactory->CreateMessage(headerStackF.size(), fair::mq::Alignment{64});
    auto plMessageF = fmqFactory->CreateMessage(hdrF.payloadSize, fair::mq::Alignment{64});
    memcpy(hdMessageF->GetData(), headerStackF.data(), headerStackF.size());
    memcpy(plMessageF->GetData(), parts.At(1)->GetData(), hdrF.payloadSize);

    fair::mq::Parts outParts;
    outParts.AddPart(std::move(hdMessageF));
    outParts.AddPart(std::move(plMessageF));
    sendOnChannel(device, outParts, channel, *timesliceId);
    LOG(info) << "Sent CTP counters DPL message" << std::flush;
  };
}

// we need to add workflow options before including Framework/runDataProcessing
void customize(std::vector<ConfigParamSpec>& workflowOptions)
{
  workflowOptions.push_back(ConfigParamSpec{"subscribe-to", VariantType::String, "type=sub,method=connect,address=tcp://188.184.30.57:5556,rateLogging=10,transport=zeromq", {"channel subscribe to"}});
  workflowOptions.push_back(ConfigParamSpec{"for-qc", VariantType::Bool, false, {"disable processing messages in run manager"}});
  workflowOptions.push_back(ConfigParamSpec{"ccdb-host", VariantType::String, "http://o2-ccdb.internal:8080", {"ccdb host"}});
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
  bool qc = config.options().get<bool>("for-qc");
  std::string ccdbhost = config.options().get<std::string>("ccdb-host");
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
    // this is just default, can be overriden by --ctp-config-proxy '--channel-config..'
    chan.c_str(),
    dcs2dpl(qc, ccdbhost));
  LOG(info) << "===> Proxy done";
  WorkflowSpec workflow;
  workflow.emplace_back(ctpProxy);
  return workflow;
}
