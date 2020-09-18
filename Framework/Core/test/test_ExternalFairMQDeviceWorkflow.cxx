// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#include "Framework/ExternalFairMQDeviceProxy.h"

using namespace o2::framework;

#include "Framework/AlgorithmSpec.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/ChannelSpec.h"
#include "Framework/DataSpecUtils.h"
#include "Framework/ExternalFairMQDeviceProxy.h"
#include "Framework/ControlService.h"
#include "Framework/Logger.h"
#include "Headers/DataHeader.h"
#include "fairmq/FairMQDevice.h"

// we need to add workflow options before including Framework/runDataProcessing
void customize(std::vector<ConfigParamSpec>& workflowOptions)
{
  workflowOptions.push_back(
    ConfigParamSpec{
      "default-transport", VariantType::String, "shmem", {"default transport: shmem, zeromq"}});
  workflowOptions.push_back(
    ConfigParamSpec{
      "number-of-events,n", VariantType::Int, 10, {"number of events to process"}});
}

#include "Framework/runDataProcessing.h"

#define ASSERT_ERROR(condition)                                   \
  if ((condition) == false) {                                     \
    LOG(ERROR) << R"(Test condition ")" #condition R"(" failed)"; \
  }

std::vector<DataProcessorSpec> defineDataProcessing(ConfigContext const& config)
{
  std::string defaultTransportConfig = config.options().get<std::string>("default-transport");
  int nRolls = config.options().get<int>("number-of-events");
  if (defaultTransportConfig == "zeromq") {
    // nothing to do for the moment
  } else if (defaultTransportConfig == "shmem") {
    // nothing to do for the moment
  } else {
    throw std::runtime_error("invalid argument for option --default-transport : '" + defaultTransportConfig + "'");
  }
  std::vector<DataProcessorSpec> workflow;

  //////////////////////////////////////////////////////////////////////////////////////////////////////////////
  // a producer process steered by a timer
  //
  // the compute callback of the producer
  auto producerCallback = [nRolls, counter = std::make_shared<int>()](DataAllocator& outputs, ControlService& control) {
    outputs.make<int>(OutputRef{"data", 0}) = *counter;
    if (++(*counter) >= nRolls) {
      control.endOfStream();
      control.readyToQuit(QuitRequest::Me);
    }
  };

  workflow.emplace_back(DataProcessorSpec{"producer",
                                          {InputSpec{"timer", "TST", "TIMER", 0, Lifetime::Timer}},
                                          {OutputSpec{{"data"}, "TST", "DATA", 0, Lifetime::Timeframe}},
                                          AlgorithmSpec{adaptStateless(producerCallback)},
                                          {ConfigParamSpec{"period-timer", VariantType::Int, 100000, {"period of timer"}}}});

  //////////////////////////////////////////////////////////////////////////////////////////////////////////////
  // the dpl sink proxy process

  // use the OutputChannelSpec as a tool to create the default configuration for the out-of-band channel
  OutputChannelSpec externalChannelSpec;
  // Note: the name has to match the binding of the input spec
  externalChannelSpec.name = "external";
  externalChannelSpec.type = ChannelType::Push;
  externalChannelSpec.method = ChannelMethod::Bind;
  externalChannelSpec.hostname = "localhost";
  externalChannelSpec.port = 42042;
  externalChannelSpec.listeners = 0;
  if (!defaultTransportConfig.empty()) {
    if (defaultTransportConfig == "zeromq") {
      externalChannelSpec.protocol = ChannelProtocol::Network;
    } else if (defaultTransportConfig == "shmem") {
      externalChannelSpec.protocol = ChannelProtocol::IPC;
    }
  }
  std::string channelConfig = formatExternalChannelConfiguration(externalChannelSpec);
  // at some point the formatting tool might add the transport as well so we have to check
  if (!defaultTransportConfig.empty() && defaultTransportConfig.find("transport=") == std::string::npos) {
    channelConfig += ",transport=" + defaultTransportConfig;
  }

  Inputs sinkInputs = {InputSpec{"external", "TST", "DATA", 0, Lifetime::Timeframe}};
  workflow.emplace_back(std::move(specifyFairMQDeviceOutputProxy("dpl-sink", sinkInputs, channelConfig.c_str())));

  //////////////////////////////////////////////////////////////////////////////////////////////////////////////
  // a simple checker process subscribing to the output of the input proxy
  //
  // the compute callback of the checker
  auto checkerCallback = [nRolls](InputRecord& inputs, ControlService& control) {
    LOG(DEBUG) << "got inputs " << inputs.size();
    if (inputs.get<int>("datain") == nRolls - 1) {
      LOG(INFO) << "terminating after " << nRolls << " successful event(s)";
      control.endOfStream();
      control.readyToQuit(QuitRequest::All);
    }
  };

  // the checker process connects to the proxy
  workflow.emplace_back(DataProcessorSpec{"checker",
                                          {InputSpec{"datain", "PRX", "DATA", 0, Lifetime::Timeframe}},
                                          {},
                                          AlgorithmSpec{adaptStateless(checkerCallback)}});

  //////////////////////////////////////////////////////////////////////////////////////////////////////////////
  // the input proxy process
  // reads the messages from the output proxy via the out-of-band channel

  // converter callback for the external FairMQ device proxy ProcessorSpec generator
  auto converter = [](FairMQDevice& device, FairMQParts& inputs, ChannelRetriever channelRetriever) {
    ASSERT_ERROR(inputs.Size() >= 2);
    if (inputs.Size() < 2) {
      return;
    }
    int msgidx = 0;
    auto dh = o2::header::get<o2::header::DataHeader*>(inputs.At(msgidx)->GetData());
    if (!dh) {
      LOG(ERROR) << "data on input " << msgidx << " does not follow the O2 data model, DataHeader missing";
      return;
    }
    auto dph = o2::header::get<DataProcessingHeader*>(inputs.At(msgidx)->GetData());
    if (!dph) {
      LOG(ERROR) << "data on input " << msgidx << " does not follow the O2 data model, DataProcessingHeader missing";
      return;
    }
    // Note: we want to run both the output and input proxy in the same workflow and thus we need
    // different data identifiers and change the data origin in the forwarding
    OutputSpec query{"PRX", dh->dataDescription, dh->subSpecification};
    auto channelName = channelRetriever(query, dph->startTime);
    ASSERT_ERROR(!channelName.empty());
    LOG(DEBUG) << "using channel '" << channelName << "' for " << DataSpecUtils::describe(OutputSpec{dh->dataOrigin, dh->dataDescription, dh->subSpecification});
    if (channelName.empty()) {
      return;
    }
    // make a copy of the header message, get the data header and change origin
    auto outHeaderMessage = device.NewMessageFor(channelName, 0, inputs.At(msgidx)->GetSize());
    memcpy(outHeaderMessage->GetData(), inputs.At(msgidx)->GetData(), inputs.At(msgidx)->GetSize());
    // this we obviously need to fix in the get API, const'ness of the returned header pointer
    // should depend on const'ness of the buffer
    auto odh = const_cast<o2::header::DataHeader*>(o2::header::get<o2::header::DataHeader*>(outHeaderMessage->GetData()));
    odh->dataOrigin = o2::header::DataOrigin("PRX");
    FairMQParts output;
    output.AddPart(std::move(outHeaderMessage));
    output.AddPart(std::move(inputs.At(msgidx + 1)));
    LOG(DEBUG) << "sending " << DataSpecUtils::describe(OutputSpec{odh->dataOrigin, odh->dataDescription, odh->subSpecification});
    o2::framework::sendOnChannel(device, output, channelName);
  };

  // we use the same spec to build the configuration string, ideally we would have some helpers
  // which convert an OutputChannelSpec to an InputChannelSpec replacing 'bind' <--> 'connect'
  // and 'push' <--> 'pull'
  //
  // skip the name in the configuration string as it is added in specifyExternalFairMQDeviceProxy
  externalChannelSpec.name = "";
  externalChannelSpec.type = ChannelType::Pull;
  externalChannelSpec.method = ChannelMethod::Connect;
  channelConfig = formatExternalChannelConfiguration(externalChannelSpec);
  if (!defaultTransportConfig.empty() && defaultTransportConfig.find("transport=") == std::string::npos) {
    channelConfig += ",transport=" + defaultTransportConfig;
  }

  // Note: in order to make the DPL output proxy and an input proxy working in the same
  // workflow, we use different data description
  Outputs inputProxyOutputs = {OutputSpec{"PRX", "DATA", 0, Lifetime::Timeframe}};
  workflow.emplace_back(specifyExternalFairMQDeviceProxy(
    "input-proxy",
    std::move(inputProxyOutputs),
    channelConfig.c_str(),
    converter));

  return workflow;
}
