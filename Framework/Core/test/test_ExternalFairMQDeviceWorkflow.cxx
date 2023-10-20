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
#include "Framework/ExternalFairMQDeviceProxy.h"

using namespace o2::framework;

#include "Framework/AlgorithmSpec.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/ChannelSpec.h"
#include "Framework/DeviceSpec.h"
#include "Framework/DataSpecUtils.h"
#include "Framework/SourceInfoHeader.h"
#include "Framework/ExternalFairMQDeviceProxy.h"
#include "Framework/ControlService.h"
#include "Framework/CallbackService.h"
#include "Framework/RawDeviceService.h"
#include "Framework/Logger.h"
#include "Framework/InputRecordWalker.h"
#include "Headers/DataHeader.h"
#include <fairmq/Device.h>

namespace test_config
{
enum struct ProxyMode {
  All,
  SkipOutput,
  OnlyOutput, // also excludes checker
  NoProxies,
};
}

namespace test_header
{
struct MsgModeHeader : public o2::header::BaseHeader {
  enum struct MsgMode {
    Pair,
    Sequence,
  };

  static constexpr uint32_t sVersion = 1;
  static constexpr o2::header::HeaderType sHeaderType = "MsgMode";
  MsgModeHeader(MsgMode _mode, size_t nParts)
    : BaseHeader(sizeof(MsgModeHeader), sHeaderType, o2::header::gSerializationMethodNone, sVersion), mode(_mode), nPayloadParts(nParts)
  {
  }

  MsgMode mode;
  size_t nPayloadParts;
};
} // namespace test_header
std::istream& operator>>(std::istream& in, enum test_config::ProxyMode& val);
std::ostream& operator<<(std::ostream& out, const enum test_config::ProxyMode& val);
std::istream& operator>>(std::istream& in, enum test_header::MsgModeHeader::MsgMode val);
std::ostream& operator<<(std::ostream& out, const enum test_header::MsgModeHeader::MsgMode val);

// we need to add workflow options before including Framework/runDataProcessing
void customize(std::vector<ConfigParamSpec>& workflowOptions)
{
  workflowOptions.push_back(
    ConfigParamSpec{
      "default-transport", VariantType::String, "shmem", {"default transport: shmem, zeromq"}});
  workflowOptions.push_back(
    ConfigParamSpec{
      "number-of-events,n", VariantType::Int, 10, {"number of events to process"}});
  workflowOptions.push_back(
    ConfigParamSpec{
      "proxy-mode", VariantType::String, "skip-output", {"proxy mode: all, skip-output, only-output, skip-all"}});
}

#include "Framework/runDataProcessing.h"

using namespace o2::framework;
using DataHeader = o2::header::DataHeader;
using Stack = o2::header::Stack;

#define ASSERT_ERROR(condition)                                   \
  if ((condition) == false) {                                     \
    LOG(fatal) << R"(Test condition ")" #condition R"(" failed)"; \
  }

#define ASSERT_EQUAL(left, right)                                                            \
  if ((left == right) == false) {                                                            \
    LOGP(fatal, R"(Test condition {} ({}) == {} ({}) failed")", #left, left, #right, right); \
  }

template <typename T>
T readConfig(ConfigContext const& config, const char* key)
{
  auto p = config.options().get<std::string>(key);
  std::stringstream cs(p);
  T val;
  cs >> val;
  if (cs.fail()) {
    throw std::runtime_error("invalid configuration parameter '" + p + "' for key " + key);
  }
  return val;
}

std::vector<DataProcessorSpec> defineDataProcessing(ConfigContext const& config)
{
  using ProxyMode = test_config::ProxyMode;
  auto proxyMode = readConfig<ProxyMode>(config, "proxy-mode");
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
  // configuration of the out-of-band proxy channel
  //
  // used either in the output proxy ('dpl-sink') or as a direct channel of the producer
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
  externalChannelSpec.sendBufferSize = 1;
  externalChannelSpec.recvBufferSize = 1;
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

  //////////////////////////////////////////////////////////////////////////////////////////////////////////////
  // a producer process steered by a timer
  //
  auto producerInitCallback = [nRolls, proxyMode, externalChannelSpec](CallbackService& callbacks, RawDeviceService& rds) {
    srand(getpid());
    auto channelName = std::make_shared<std::string>();
    auto producerChannelInit = [channelName, outputRoutes = rds.spec().outputs]() {
      // find the output channel name, we expect all output messages to be
      // sent over the same channel
      if (channelName->empty()) {
        OutputSpec const query{"TST", "DATA", 0};
        for (auto& route : outputRoutes) {
          if (DataSpecUtils::match(route.matcher, query)) {
            *channelName = route.channel;
            break;
          }
        }
      }
      ASSERT_ERROR(channelName->length() > 0);
    };
    if (proxyMode == ProxyMode::SkipOutput) {
      *channelName = externalChannelSpec.name;
    } else {
      callbacks.set<CallbackService::Id::Start>(producerChannelInit);
    }
    // the compute callback of the producer
    auto producerCallback = [nRolls, channelName, proxyMode, counter = std::make_shared<size_t>()](DataAllocator& outputs, ControlService& control, RawDeviceService& rds) {
      int data = *counter;
      // outputs.make<int>(OutputRef{"data", 0}) = data;

      fair::mq::Device& device = *(rds.device());
      auto transport = device.GetChannel(*channelName, 0).Transport();
      auto channelAlloc = o2::pmr::getTransportAllocator(transport);

      DataProcessingHeader dph{*counter, 0};

      auto msgMode = rand() % 2 ? test_header::MsgModeHeader::MsgMode::Pair : test_header::MsgModeHeader::MsgMode::Sequence;
      size_t nPayloads = rand() % 10 + 1;

      test_header::MsgModeHeader mmh{msgMode, nPayloads};
      fair::mq::Parts messages;
      auto insertHeader = [&dph, &mmh, &channelAlloc, &messages](DataHeader const& dh) -> void {
        fair::mq::MessagePtr header = o2::pmr::getMessage(Stack{channelAlloc, dh, dph, mmh});
        messages.AddPart(std::move(header));
      };
      auto insertPayload = [&transport, &messages, &data](size_t size) -> void {
        fair::mq::MessagePtr payload = transport->CreateMessage(size);
        memcpy(payload->GetData(), &data, sizeof(data));
        messages.AddPart(std::move(payload));
      };
      auto createSequence = [&insertHeader, &insertPayload, &data](size_t nPayloads, DataHeader dh) -> void {
        // one header with index set to the number of split parts indicates sequence
        // of payloads without additional headers
        dh.payloadSize = sizeof(data);
        dh.payloadSerializationMethod = o2::header::gSerializationMethodNone;
        dh.splitPayloadIndex = nPayloads;
        dh.splitPayloadParts = nPayloads;
        insertHeader(dh);

        for (size_t i = 0; i < nPayloads; ++i) {
          insertPayload(dh.payloadSize);
        }
      };

      auto createPairs = [&insertHeader, &insertPayload, &data](size_t nPayloads, DataHeader dh) -> void {
        // one header with index set to the number of split parts indicates sequence
        // of payloads without additional headers
        dh.payloadSize = sizeof(data);
        dh.payloadSerializationMethod = o2::header::gSerializationMethodNone;
        dh.splitPayloadIndex = 0;
        dh.splitPayloadParts = nPayloads;
        for (size_t i = 0; i < nPayloads; ++i) {
          dh.splitPayloadIndex = i;
          insertHeader(dh);
          insertPayload(dh.payloadSize);
        }
      };

      if (msgMode == test_header::MsgModeHeader::MsgMode::Pair) {
        createPairs(nPayloads, DataHeader{"DATA", "TST", 0});
      } else {
        createSequence(nPayloads, DataHeader{"DATA", "TST", 0});
      }
      // using utility from ExternalFairMQDeviceProxy
      o2::framework::sendOnChannel(device, messages, *channelName, (size_t)-1);

      if (++(*counter) >= nRolls) {
        // send the end of stream signal, this is transferred by the proxies
        // and allows to properly terminate downstream devices
        control.endOfStream();
        if (proxyMode == ProxyMode::SkipOutput) {
          // since we are sending on the bare channel, also the EOS message needs to be created.
          DataHeader dhEOS;
          dhEOS.dataOrigin = "DPL";
          dhEOS.dataDescription = "EOS";
          dhEOS.subSpecification = 0;
          dhEOS.payloadSize = 0;
          dhEOS.payloadSerializationMethod = o2::header::gSerializationMethodNone;
          dhEOS.tfCounter = 0;
          dhEOS.firstTForbit = 0;
          SourceInfoHeader sih;
          sih.state = InputChannelState::Completed;
          auto headerMessage = o2::pmr::getMessage(o2::header::Stack{channelAlloc, dhEOS, dph, sih});
          fair::mq::Parts out;
          out.AddPart(std::move(headerMessage));
          // add empty payload message
          out.AddPart(std::move(device.NewMessageFor(*channelName, 0, 0)));
          o2::framework::sendOnChannel(device, out, *channelName, (size_t)-1);
        }
      }
    };
    return adaptStateless(producerCallback);
  };

  workflow.emplace_back(DataProcessorSpec{"producer",
                                          {InputSpec{"timer", "TST", "TIMER", 0, Lifetime::Timer}},
                                          {OutputSpec{{"data"}, "TST", "DATA", 0, Lifetime::Timeframe}},
                                          AlgorithmSpec{adaptStateful(producerInitCallback)},
                                          {ConfigParamSpec{"period-timer", VariantType::Int, 100000, {"period of timer"}}}});

  if (proxyMode == ProxyMode::SkipOutput) {
    // create the out-of-band channel in the producer if the output proxy is bypassed
    const char* d = strdup(channelConfig.c_str());
    workflow.back().options.push_back(ConfigParamSpec{"channel-config", VariantType::String, d, {"proxy channel of producer"}});
  }

  //////////////////////////////////////////////////////////////////////////////////////////////////////////////
  // the dpl sink proxy process

  Inputs sinkInputs = {InputSpec{"external", "TST", "DATA", 0, Lifetime::Timeframe}};
  auto channelSelector = [](InputSpec const&, const std::unordered_map<std::string, std::vector<fair::mq::Channel>>&) -> std::string {
    return "downstream";
  };
  if (proxyMode == ProxyMode::All || proxyMode == ProxyMode::OnlyOutput) {
    workflow.emplace_back(std::move(specifyFairMQDeviceMultiOutputProxy("dpl-sink", sinkInputs, channelConfig.c_str(), channelSelector)));
  }

  //////////////////////////////////////////////////////////////////////////////////////////////////////////////
  // a simple checker process subscribing to the output of the input proxy
  //
  // the compute callback of the checker
  auto counter = std::make_shared<int>(0);
  auto checkerCallback = [counter](InputRecord& inputs, ControlService& control) {
    auto const* dh = DataRefUtils::getHeader<o2::header::DataHeader*>(inputs.get("datain"));
    auto const* mmh = DataRefUtils::getHeader<test_header::MsgModeHeader*>(inputs.get("datain"));
    ASSERT_ERROR(dh != nullptr);
    ASSERT_ERROR(mmh != nullptr);
    LOGP(info, "{} input slots(s), data {}, parts {}, mode {}", inputs.size(), inputs.get<int>("datain"), mmh->nPayloadParts, (int)mmh->mode);
    if (mmh->mode == test_header::MsgModeHeader::MsgMode::Pair) {
      ASSERT_ERROR(dh->splitPayloadParts == mmh->nPayloadParts);
      ASSERT_ERROR(dh->splitPayloadIndex == 0);
    } else {
      ASSERT_ERROR(dh->splitPayloadParts == mmh->nPayloadParts);
      ASSERT_ERROR(dh->splitPayloadIndex == mmh->nPayloadParts);
    }
    size_t nPayloads = 0;
    for (auto const& ref : InputRecordWalker(inputs)) {
      auto data = inputs.get<int>(ref);
      ASSERT_ERROR(data == *counter);
      ++nPayloads;
    }
    ASSERT_ERROR(nPayloads == mmh->nPayloadParts);
    ++(*counter);
  };
  auto checkCounter = [counter, nRolls](EndOfStreamContext&) {
    ASSERT_EQUAL(*counter, nRolls);
    if (*counter == nRolls) {
      LOG(info) << "checker has received " << nRolls << " successful event(s)";
    }
  };
  auto checkerInit = [checkerCallback, checkCounter](CallbackService& callbacks) {
    callbacks.set<CallbackService::Id::EndOfStream>(checkCounter);
    return adaptStateless(checkerCallback);
  };

  // the checker process connects to the proxy
  Inputs checkerInputs;
  if (proxyMode != ProxyMode::All) {
    checkerInputs.emplace_back(InputSpec{"datain", ConcreteDataTypeMatcher{"TST", "DATA"}, Lifetime::Timeframe});
    // for (unsigned int i = 0; i < pState->nChannels; i++) {
    //   checkerInputs.emplace_back(InputSpec{{"datain"}, "TST", "DATA", i, Lifetime::Timeframe});
    // }
  } else {
    checkerInputs.emplace_back(InputSpec{"datain", ConcreteDataTypeMatcher{"PRX", "DATA"}, Lifetime::Timeframe});
    // for (unsigned int i = 0; i < pState->nChannels; i++) {
    //   checkerInputs.emplace_back(InputSpec{{"datain"}, "PRX", "DATA", i, Lifetime::Timeframe});
    // }
  }
  if (proxyMode != ProxyMode::OnlyOutput) {
    // the checker is not added if the input proxy is skipped
    workflow.emplace_back(DataProcessorSpec{"checker",
                                            std::move(checkerInputs),
                                            {},
                                            AlgorithmSpec{adaptStateful(checkerInit)}});
  }

  //////////////////////////////////////////////////////////////////////////////////////////////////////////////
  // the input proxy process
  // reads the messages from the output proxy via the out-of-band channel

  // converter callback for the external FairMQ device proxy ProcessorSpec generator
  InjectorFunction converter = [](TimingInfo&, ServiceRegistryRef const& services, fair::mq::Parts& inputs, ChannelRetriever channelRetriever, size_t newTimesliceId, bool&) -> bool {
    auto* device = services.get<RawDeviceService>().device();
    ASSERT_ERROR(inputs.Size() >= 2);
    if (inputs.Size() < 2) {
      return false;
    }
    int msgidx = 0;
    auto dh = o2::header::get<o2::header::DataHeader*>(inputs.At(msgidx)->GetData());
    if (!dh) {
      LOG(error) << "data on input " << msgidx << " does not follow the O2 data model, DataHeader missing";
      return false;
    }
    auto dph = o2::header::get<DataProcessingHeader*>(inputs.At(msgidx)->GetData());
    if (!dph) {
      LOG(error) << "data on input " << msgidx << " does not follow the O2 data model, DataProcessingHeader missing";
      return false;
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
    LOG(debug) << "using channel '" << channelName << "' for " << DataSpecUtils::describe(OutputSpec{dh->dataOrigin, dh->dataDescription, dh->subSpecification});
    if (channelName.empty()) {
      return false;
    }
    fair::mq::Parts output;
    for (; msgidx < inputs.Size(); ++msgidx) {
      auto const* dh = o2::header::get<o2::header::DataHeader*>(inputs.At(msgidx)->GetData());
      if (dh) {
        LOGP(debug, "{}/{}/{} with {} part(s), index {}",
             dh->dataOrigin.as<std::string>(),
             dh->dataDescription.as<std::string>(),
             dh->subSpecification,
             dh->splitPayloadParts,
             dh->splitPayloadIndex);
        // make a copy of the header message, get the data header and change origin
        auto outHeaderMessage = device->NewMessageFor(channelName, 0, inputs.At(msgidx)->GetSize());
        memcpy(outHeaderMessage->GetData(), inputs.At(msgidx)->GetData(), inputs.At(msgidx)->GetSize());
        // this we obviously need to fix in the get API, const'ness of the returned header pointer
        // should depend on const'ness of the buffer
        auto odh = const_cast<o2::header::DataHeader*>(o2::header::get<o2::header::DataHeader*>(outHeaderMessage->GetData()));
        odh->dataOrigin = o2::header::DataOrigin("PRX");
        output.AddPart(std::move(outHeaderMessage));
      } else {
        output.AddPart(std::move(inputs.At(msgidx)));
      }
    }
    o2::framework::sendOnChannel(*device, output, channelName, (size_t)-1);
    return output.Size() != 0;
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

  if (proxyMode == ProxyMode::All) {
    // Note: in order to make the DPL output proxy and an input proxy working in the same
    // workflow, we use different data description
    Outputs inputProxyOutputs = {OutputSpec{ConcreteDataTypeMatcher{"PRX", "DATA"}, Lifetime::Timeframe}};
    workflow.emplace_back(specifyExternalFairMQDeviceProxy(
      "input-proxy",
      std::move(inputProxyOutputs),
      channelConfig.c_str(),
      converter));
  } else if (proxyMode == ProxyMode::SkipOutput) {
    Outputs inputProxyOutputs = {OutputSpec{ConcreteDataTypeMatcher{"TST", "DATA"}, Lifetime::Timeframe}};
    // we use the same specs as filters in the dpl adaptor
    auto filterSpecs = inputProxyOutputs;
    workflow.emplace_back(specifyExternalFairMQDeviceProxy(
      "input-proxy",
      std::move(inputProxyOutputs),
      channelConfig.c_str(),
      o2::framework::dplModelAdaptor(filterSpecs, true)));
  }

  return workflow;
}

std::istream& operator>>(std::istream& in, enum test_config::ProxyMode& val)
{
  std::string token;
  in >> token;
  if (token == "all" || token == "a") {
    val = test_config::ProxyMode::All;
  } else if (token == "skip-output") {
    val = test_config::ProxyMode::SkipOutput;
  } else if (token == "only-output") {
    val = test_config::ProxyMode::OnlyOutput;
  } else if (token == "skip-all" || token == "skip-proxies") {
    val = test_config::ProxyMode::NoProxies;
  } else {
    in.setstate(std::ios_base::failbit);
  }
  return in;
}

std::ostream& operator<<(std::ostream& out, const enum test_config::ProxyMode& val)
{
  if (val == test_config::ProxyMode::All) {
    out << "all";
  } else if (val == test_config::ProxyMode::SkipOutput) {
    out << "skip-output";
  } else if (val == test_config::ProxyMode::OnlyOutput) {
    out << "only-output";
  } else if (val == test_config::ProxyMode::NoProxies) {
    out << "skip-all";
  } else {
    out.setstate(std::ios_base::failbit);
  }
  return out;
}

std::istream& operator>>(std::istream& in, enum test_header::MsgModeHeader::MsgMode& val)
{
  std::string token;
  in >> token;
  if (token == "pair") {
    val = test_header::MsgModeHeader::MsgMode::Pair;
  } else if (token == "sequence") {
    val = test_header::MsgModeHeader::MsgMode::Sequence;
  } else {
    in.setstate(std::ios_base::failbit);
  }
  return in;
}

std::ostream& operator<<(std::ostream& out, const enum test_header::MsgModeHeader::MsgMode& val)
{
  if (val == test_header::MsgModeHeader::MsgMode::Pair) {
    out << "pair";
  } else if (val == test_header::MsgModeHeader::MsgMode::Sequence) {
    out << "sequence";
  } else {
    out.setstate(std::ios_base::failbit);
  }
  return out;
}
