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
#include "Framework/CallbackService.h"
#include "Framework/Logger.h"
#include "Framework/InputRecordWalker.h"
#include "Headers/DataHeader.h"
#include "fairmq/FairMQDevice.h"
#include <chrono>

// we need to add workflow options before including Framework/runDataProcessing
void customize(std::vector<ConfigParamSpec>& workflowOptions)
{
  workflowOptions.push_back(
    ConfigParamSpec{
      "default-transport", VariantType::String, "shmem", {"default transport: shmem, zeromq"}});
  workflowOptions.push_back(
    ConfigParamSpec{
      "nChannels", VariantType::Int, 1, {"number of output channels of the producer"}});
  workflowOptions.push_back(
    ConfigParamSpec{
      "msgSize", VariantType::Int, 1024, {"message size in kB"}});
  workflowOptions.push_back(
    ConfigParamSpec{
      "runningTime", VariantType::Int, 30, {"time to run the workflow"}});
}

#include "Framework/runDataProcessing.h"

using benchclock = std::chrono::high_resolution_clock;

#define ASSERT_ERROR(condition)                                   \
  if ((condition) == false) {                                     \
    LOG(FATAL) << R"(Test condition ")" #condition R"(" failed)"; \
  }

std::vector<DataProcessorSpec> defineDataProcessing(ConfigContext const& config)
{
  std::string defaultTransportConfig = config.options().get<std::string>("default-transport");
  if (defaultTransportConfig == "zeromq") {
    // nothing to do for the moment
  } else if (defaultTransportConfig == "shmem") {
    // nothing to do for the moment
  } else {
    throw std::runtime_error("invalid argument for option --default-transport : '" + defaultTransportConfig + "'");
  }
  std::vector<DataProcessorSpec> workflow;

  struct BenchmarkState {
    size_t msgSize = 1024 * 1024;
    size_t nChannels = 1;
    size_t logPeriod = 2;
    size_t runningTime = 30;
    size_t counter = 0;
    size_t totalCount = 0;
    benchclock::time_point startTime = benchclock::now();
    benchclock::time_point idleTime = benchclock::now();
    benchclock::time_point lastLogTime = benchclock::now();
    float maxMsgPerSec = .0;
    float maxDataRatePerSec = .0;
    float totalIdleTime = .0;
    float maxIdleTime = .0;
  };

  auto makeBenchmarkState = [&config]() -> std::shared_ptr<BenchmarkState> {
    auto state = std::make_shared<BenchmarkState>();
    state->msgSize = 1024 * config.options().get<int>("msgSize");
    state->nChannels = config.options().get<int>("nChannels");
    state->runningTime = config.options().get<int>("runningTime");
    return state;
  };

  auto loggerInit = [](BenchmarkState& state) {
    state.startTime = benchclock::now();
    state.idleTime = benchclock::now();
    state.lastLogTime = benchclock::now();
    state.totalIdleTime = 0.;
  };

  auto loggerCycle = [](BenchmarkState& state) {
    ++state.counter;
    auto secSinceLastLog = std::chrono::duration_cast<std::chrono::seconds>(benchclock::now() - state.lastLogTime);
    if (secSinceLastLog.count() >= state.logPeriod) {
      // TODO: introduce real counters for accumulated number of messages and message size
      state.totalCount += state.counter;
      float eventRate = state.counter / secSinceLastLog.count();
      float msgPerSec = eventRate * state.nChannels;
      float kbPerSec = msgPerSec * state.msgSize / 1024;
      auto elapsedTime = std::chrono::duration_cast<std::chrono::seconds>(benchclock::now() - state.startTime);
      LOG(INFO) << fmt::format(
        "{: 3d} Total messages: {} - Event rate {:.2f} Hz  {:.2f} msg/s  {:.2f} MB/s, "
        "Accumulated idle time {:.2f} ms",
        elapsedTime.count(), state.totalCount, eventRate, msgPerSec,
        kbPerSec / 1024, state.totalIdleTime / 1000);
      if (state.maxMsgPerSec < msgPerSec) {
        state.maxMsgPerSec = msgPerSec;
      }
      if (state.maxDataRatePerSec < kbPerSec) {
        state.maxDataRatePerSec = kbPerSec;
      }
      state.counter = 0;
      state.lastLogTime = benchclock::now();
    }
  };

  struct ActiveGuard {
    ActiveGuard(BenchmarkState& _state) : state(_state)
    {
      auto idleTime = std::chrono::duration_cast<std::chrono::microseconds>(benchclock::now() - state.idleTime);
      state.totalIdleTime += idleTime.count();
    }
    ~ActiveGuard()
    {
      state.idleTime = benchclock::now();
    }
    BenchmarkState& state;
  };

  auto loggerSummary = [](BenchmarkState& state) {
    auto totalTime = std::chrono::duration_cast<std::chrono::seconds>(benchclock::now() - state.startTime);
    float eventRate = state.totalCount / totalTime.count();
    float msgPerSec = eventRate * state.nChannels;
    float kbPerSec = msgPerSec * state.msgSize / 1024;
    LOG(INFO) << fmt::format(
      "Benchmarking "
#ifndef NDEBUG
      "accumulated "
#endif
      "for {} s:  Avrg event rate {:.2f} Hz, "
      "Avrg message rate {:.2f}/s (max {:.2f}/s), Avrg data rate {:.2f} MB/s (max {:.2f} MB/s), "
      "Avrg idle time {:.2f} ms",
      totalTime.count(), eventRate,
      msgPerSec, state.maxMsgPerSec,
      kbPerSec / 1024, state.maxDataRatePerSec / 1024,
      state.totalIdleTime / (state.totalCount * 1000));
  };

  //////////////////////////////////////////////////////////////////////////////////////////////////////////////
  // a producer process steered by a timer
  //
  // the compute callback of the producer
  auto pState = makeBenchmarkState();
  auto producerInitCallback = [pState, loggerInit, loggerCycle, loggerSummary](CallbackService& callbacks) {
    auto producerBenchInit = [pState, loggerInit]() {
      loggerInit(*pState);
    };
    callbacks.set(CallbackService::Id::Start, producerBenchInit);
    auto producerCallback = [pState, loggerCycle, loggerSummary](DataAllocator& outputs, ControlService& control) {
      auto& state = *pState;
      ActiveGuard g(state);
      for (unsigned int i = 0; i < state.nChannels; i++) {
        outputs.make<char>(OutputRef{"data", i}, state.msgSize);
      }
      loggerCycle(*pState);
      auto elapsedTime = std::chrono::duration_cast<std::chrono::seconds>(benchclock::now() - state.startTime);
      if (elapsedTime.count() >= state.runningTime) {
        loggerSummary(*pState);
        // send the end of stream signal, this is transferred by the proxies
        // and allows to properly terminate downstream devices
        control.endOfStream();
      }
    };

    return adaptStateless(producerCallback);
  };

  Outputs outputs;
  for (unsigned int i = 0; i < pState->nChannels; i++) {
    outputs.emplace_back(OutputSpec{{"data"}, "TST", "DATA", i, Lifetime::Timeframe});
  }
  workflow.emplace_back(DataProcessorSpec{"producer",
                                          {},
                                          {std::move(outputs)},
                                          AlgorithmSpec{adaptStateful(producerInitCallback)},
                                          {}});

  //////////////////////////////////////////////////////////////////////////////////////////////////////////////
  // the dpl sink proxy process

  // use the OutputChannelSpec as a tool to create the default configuration for the out-of-band channel
  OutputChannelSpec externalChannelSpec;
  // Note: the name is hardcoded for now
  externalChannelSpec.name = "downstream";
  externalChannelSpec.type = ChannelType::Push;
  externalChannelSpec.method = ChannelMethod::Bind;
  externalChannelSpec.hostname = "localhost";
  externalChannelSpec.port = 42042;
  externalChannelSpec.listeners = 0;
  externalChannelSpec.rateLogging = 10;
  externalChannelSpec.sendBufferSize = 1000;
  externalChannelSpec.recvBufferSize = 1000;
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

  Inputs sinkInputs;
  for (unsigned int i = 0; i < pState->nChannels; i++) {
    sinkInputs.emplace_back(InputSpec{{"external"}, "TST", "DATA", i, Lifetime::Timeframe});
  }
  auto channelSelector = [](InputSpec const&, const std::unordered_map<std::string, std::vector<FairMQChannel>>&) -> std::string {
    return "downstream";
  };
  workflow.emplace_back(std::move(specifyFairMQDeviceOutputProxy("dpl-sink", sinkInputs, channelConfig.c_str())));

  //////////////////////////////////////////////////////////////////////////////////////////////////////////////
  // a simple checker process subscribing to the output of the input proxy
  //
  // the compute callback of the checker
  auto cState = makeBenchmarkState();
  auto checkerCallback = [cState, loggerCycle](InputRecord& inputs) {
    ActiveGuard g(*cState);
    LOG(DEBUG) << "got inputs " << inputs.size();
    for (auto const& ref : InputRecordWalker(inputs)) {
      auto data = inputs.get<gsl::span<char>>(ref);
    }
    loggerCycle(*cState);
  };
  auto checkerBenchInit = [cState, loggerInit]() {
    loggerInit(*cState);
  };
  auto checkerBenchSummary = [cState, loggerSummary](EndOfStreamContext&) {
    loggerSummary(*cState);
  };
  auto checkerInit = [checkerCallback, checkerBenchInit, checkerBenchSummary](CallbackService& callbacks) {
    callbacks.set(CallbackService::Id::Start, checkerBenchInit);
    callbacks.set(CallbackService::Id::EndOfStream, checkerBenchSummary);
    return adaptStateless(checkerCallback);
  };

  // the checker process connects to the proxy
  workflow.emplace_back(DataProcessorSpec{"checker",
                                          {InputSpec{"datain", ConcreteDataTypeMatcher{"PRX", "DATA"}, Lifetime::Timeframe}},
                                          {},
                                          AlgorithmSpec{adaptStateful(checkerInit)}});

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
    bool isData = DataSpecUtils::match(OutputSpec{"TST", "DATA", 0}, dh->dataOrigin, dh->dataDescription, dh->subSpecification);
    // for the configured data channel we require the channel name, the EOS message containing
    // the forwarded SourceInfoHeader created by the output proxy will be skipped here since the
    // input proxy handles this internally
    ASSERT_ERROR(!isData || !channelName.empty());
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
  Outputs inputProxyOutputs = {OutputSpec{ConcreteDataTypeMatcher{"PRX", "DATA"}, Lifetime::Timeframe}};
  workflow.emplace_back(specifyExternalFairMQDeviceProxy(
    "input-proxy",
    std::move(inputProxyOutputs),
    channelConfig.c_str(),
    converter));

  return workflow;
}
