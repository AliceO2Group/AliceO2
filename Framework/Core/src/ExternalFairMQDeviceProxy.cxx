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
#include "Framework/AlgorithmSpec.h"
#include "Framework/ConfigParamSpec.h"
#include "Framework/AlgorithmSpec.h"
#include "Framework/DataProcessingHeader.h"
#include "Framework/DataSpecUtils.h"
#include "Framework/DeviceSpec.h"
#include "Framework/ExternalFairMQDeviceProxy.h"
#include "Framework/InitContext.h"
#include "Framework/ProcessingContext.h"
#include "Framework/RawDeviceService.h"
#include "Framework/CallbackService.h"
#include "Framework/ControlService.h"
#include "Framework/SourceInfoHeader.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/RateLimiter.h"
#include "Framework/TimingInfo.h"
#include "Headers/DataHeader.h"
#include "Headers/Stack.h"

#include "./DeviceSpecHelpers.h"
#include "Framework/DataProcessingHelpers.h"

#include <fairmq/FairMQParts.h>
#include <fairmq/FairMQDevice.h>
#include <cstring>
#include <cassert>
#include <memory>
#include <optional>
#include <unordered_map>
#include <numeric> // std::accumulate
#include <sstream>
#include <stdexcept>
#include <regex>

namespace o2::framework
{

using DataHeader = o2::header::DataHeader;

std::string formatExternalChannelConfiguration(InputChannelSpec const& spec)
{
  return DeviceSpecHelpers::inputChannel2String(spec);
}

std::string formatExternalChannelConfiguration(OutputChannelSpec const& spec)
{
  return DeviceSpecHelpers::outputChannel2String(spec);
}

std::string formatExternalChannelConfiguration(OutputChannelSpec const&);

void sendOnChannel(FairMQDevice& device, FairMQParts& messages, std::string const& channel, size_t timeSlice)
{
  // Note: DPL is only setting up one instance of a channel while FairMQ allows to have an
  // array of channels, the index is 0 in the call
  constexpr auto index = 0;
  LOG(debug) << "sending " << messages.Size() << " messages on " << channel;
  // TODO: we can make this configurable
  const int maxTimeout = 10000;
  int timeout = 0;
  // try dispatch with increasing timeout in order to also drop a warning if the dispatching
  // has been tried multiple times within max timeout
  // since we do not want any messages to be dropped at this stage, we stay in the loop until
  // the downstream congestion is resolved
  // TODO: we might want to treat this error condition some levels higher up, but for
  // the moment its an appropriate solution. The important thing is not to drop
  // messages and to be informed about the congestion.
  while (device.Send(messages, channel, index, timeout) < 0) {
    if (timeout == 0) {
      timeout = 1;
    } else if (timeout < maxTimeout) {
      timeout *= 10;
    } else {
      LOG(error) << "failed to dispatch messages on channel " << channel << ", downstream queue might be full\n"
                 << "or unconnected. No data is dropped, keep on trying, but this will hold the reading from\n"
                 << "the input and expose back-pressure upstream. RESOLVE DOWNSTREAM CONGESTION to continue";
      if (timeout == maxTimeout) {
        // we add 1ms to disable the warning below
        timeout += 1;
      }
    }
    if (device.NewStatePending()) {
      LOG(error) << "device state change is requested, dropping " << messages.Size() << " pending message(s)\n"
                 << "on channel " << channel << "\n"
                 << "ATTENTION: DATA IS LOST! Could not dispatch data to downstream consumer(s), check if\n"
                 << "consumers have been terminated too early";
      // make sure we disable the warning below
      timeout = maxTimeout + 1;
      break;
    }
  }
  
  // FIXME: we need a better logic for avoiding message spam
  if (timeout > 1 && timeout <= maxTimeout) {
    LOG(warning) << "dispatching on channel " << channel << " was delayed by " << timeout << " ms";
  }
  // TODO: feeling this is a bit awkward, but the interface of FairMQParts does not provide a
  // method to clear the content.
  // Maybe the FairMQ API can be improved at some point. Actually the ownership of all messages should be passed
  // on to the transport and the messages should be empty after sending and the parts content can be cleared.
  // assert(std::accumulate(messages.begin(), messages.end(), true, [](bool a, auto const& msg) {return a && (msg.get() == nullptr);}));
  messages.fParts.clear();
}

void sendOnChannel(FairMQDevice& device, FairMQParts& messages, OutputSpec const& spec, DataProcessingHeader::StartTime tslice, ChannelRetriever& channelRetriever)
{
  // Note: DPL is only setting up one instance of a channel while FairMQ allows to have an
  // array of channels, the index is 0 in the call
  auto channel = channelRetriever(spec, tslice);
  if (channel.empty()) {
    LOG(warning) << "can not find matching channel for " << DataSpecUtils::describe(spec) << " timeslice " << tslice;
    return;
  }
  sendOnChannel(device, messages, channel, tslice);
}

void sendOnChannel(FairMQDevice& device, o2::header::Stack&& headerStack, FairMQMessagePtr&& payloadMessage, OutputSpec const& spec, ChannelRetriever& channelRetriever)
{
  const auto* dph = o2::header::get<DataProcessingHeader*>(headerStack.data());
  if (!dph) {
    LOG(error) << "Header Stack does not follow the O2 data model, DataProcessingHeader missing";
    return;
  }
  auto channelName = channelRetriever(spec, dph->startTime);
  constexpr auto index = 0;
  if (channelName.empty()) {
    LOG(warning) << "can not find matching channel for " << DataSpecUtils::describe(spec);
    return;
  }
  for (auto& channelInfo : device.fChannels) {
    if (channelInfo.first != channelName) {
      continue;
    }
    assert(channelInfo.second.size() == 1);
    // allocate the header message using the underlying transport of the channel
    auto channelAlloc = o2::pmr::getTransportAllocator(channelInfo.second[index].Transport());
    FairMQMessagePtr headerMessage = o2::pmr::getMessage(std::move(headerStack), channelAlloc);

    FairMQParts out;
    out.AddPart(std::move(headerMessage));
    out.AddPart(std::move(payloadMessage));
    sendOnChannel(device, out, channelName, dph->startTime);
    return;
  }
  LOG(error) << "internal mismatch, can not find channel " << channelName << " in the list of channel infos of the device";
}

void sendOnChannel(FairMQDevice& device, FairMQMessagePtr&& headerMessage, FairMQMessagePtr&& payloadMessage, OutputSpec const& spec, ChannelRetriever& channelRetriever)
{
  //  const auto* dph = o2::header::get<DataProcessingHeader*>( *reinterpret_cast<o2::header::Stack*>(headerMessage->GetData()) );
  const auto* dph = o2::header::get<DataProcessingHeader*>(headerMessage->GetData());
  if (!dph) {
    LOG(error) << "Header does not follow the O2 data model, DataProcessingHeader missing";
    return;
  }
  auto tslice = dph->startTime;
  FairMQParts out;
  out.AddPart(std::move(headerMessage));
  out.AddPart(std::move(payloadMessage));
  sendOnChannel(device, out, spec, tslice, channelRetriever);
}

InjectorFunction o2DataModelAdaptor(OutputSpec const& spec, uint64_t startTime, uint64_t /*step*/)
{
  auto timesliceId = std::make_shared<size_t>(startTime);
  return [timesliceId, spec](TimingInfo&, FairMQDevice& device, FairMQParts& parts, ChannelRetriever channelRetriever) {
    for (int i = 0; i < parts.Size() / 2; ++i) {
      auto dh = o2::header::get<DataHeader*>(parts.At(i * 2)->GetData());

      DataProcessingHeader dph{*timesliceId, 0};
      o2::header::Stack headerStack{*dh, dph};
      sendOnChannel(device, std::move(headerStack), std::move(parts.At(i * 2 + 1)), spec, channelRetriever);
      *timesliceId += 1;
    }
  };
}

InjectorFunction dplModelAdaptor(std::vector<OutputSpec> const& filterSpecs, DPLModelAdapterConfig config)
{
  bool throwOnUnmatchedInputs = config.throwOnUnmatchedInputs;
  // structure to hold information on the unmatched data and print a warning at cleanup
  class DroppedDataSpecs
  {
   public:
    DroppedDataSpecs() = default;
    ~DroppedDataSpecs()
    {
      warning();
    }

    [[nodiscard]] bool find(std::string const& desc) const
    {
      return descriptions.find(desc) != std::string::npos;
    }

    void add(std::string const& desc)
    {
      descriptions += "\n   " + desc;
    }

    void warning() const
    {
      if (not descriptions.empty()) {
        LOG(warning) << "Some input data could not be matched by filter rules to output specs\n"
                     << "Active rules: " << descriptions << "\n"
                     << "DROPPING OF THESE MESSAGES HAS BEEN ENABLED BY CONFIGURATION";
      }
    }

   private:
    std::string descriptions;
  };

  return [filterSpecs = std::move(filterSpecs), throwOnUnmatchedInputs, droppedDataSpecs = std::make_shared<DroppedDataSpecs>()](TimingInfo& timingInfo, FairMQDevice& device, FairMQParts& parts, ChannelRetriever channelRetriever) {
    std::unordered_map<std::string, FairMQParts> outputs;
    std::vector<std::string> unmatchedDescriptions;
    static int64_t dplCounter = -1;
    dplCounter++;
    for (int msgidx = 0; msgidx < parts.Size(); msgidx += 2) {
      const auto dh = o2::header::get<DataHeader*>(parts.At(msgidx)->GetData());
      if (!dh) {
        LOG(error) << "data on input " << msgidx << " does not follow the O2 data model, DataHeader missing";
        if (msgidx > 0) {
          --msgidx;
        }
        continue;
      }
      auto dph = o2::header::get<DataProcessingHeader*>(parts.At(msgidx)->GetData());
      if (!dph) {
        LOG(error) << "data on input " << msgidx << " does not follow the O2 data model, DataProcessingHeader missing";
        continue;
      }
      const_cast<DataProcessingHeader*>(dph)->startTime = dplCounter;
      timingInfo.timeslice = dph->startTime;
      timingInfo.creation = dph->creation;
      timingInfo.firstTFOrbit = dh->firstTForbit;
      timingInfo.runNumber = dh->runNumber;
      timingInfo.tfCounter = dh->tfCounter;
      LOG(debug) << msgidx << ": " << DataSpecUtils::describe(OutputSpec{dh->dataOrigin, dh->dataDescription, dh->subSpecification}) << " part " << dh->splitPayloadIndex << " of " << dh->splitPayloadParts << "  payload " << parts.At(msgidx + 1)->GetSize();

      OutputSpec query{dh->dataOrigin, dh->dataDescription, dh->subSpecification};
      LOG(debug) << "processing " << DataSpecUtils::describe(OutputSpec{dh->dataOrigin, dh->dataDescription, dh->subSpecification}) << " time slice " << dph->startTime << " part " << dh->splitPayloadIndex << " of " << dh->splitPayloadParts;
      int finalBlockIndex = 0;
      std::string channelName = "";

      for (auto const& spec : filterSpecs) {
        // filter on the specified OutputSpecs, the default value is a ConcreteDataTypeMatcher with origin and description 'any'
        if (DataSpecUtils::match(spec, OutputSpec{{header::gDataOriginAny, header::gDataDescriptionAny}}) ||
            DataSpecUtils::match(spec, query)) {
          channelName = channelRetriever(query, dph->startTime);
          if (channelName.empty()) {
            LOG(warning) << "can not find matching channel, not able to adopt " << DataSpecUtils::describe(query);
          }
          break;
        }
      }
      if (dh->splitPayloadParts > 0 && dh->splitPayloadParts == dh->splitPayloadIndex) {
        // this is indicating a sequence of payloads following the header
        // FIXME: we will probably also set the DataHeader version
        finalBlockIndex = msgidx + dh->splitPayloadParts + 1;
      } else {
        // We can consider the next splitPayloadParts as one block of messages pairs
        // because we are guaranteed they are all the same.
        // If splitPayloadParts = 0, we assume that means there is only one (header, payload)
        // pair.
        finalBlockIndex = msgidx + (dh->splitPayloadParts > 0 ? dh->splitPayloadParts : 1) * 2;
      }
      assert(finalBlockIndex >= msgidx + 2);
      if (finalBlockIndex > parts.Size()) {
        // TODO error handling
        // LOGP(error, "DataHeader::splitPayloadParts invalid");
        continue;
      }

      if (!channelName.empty()) {
        // the checks for consistency of split payload parts are of informative nature
        // forwarding happens independently
        // if (dh->splitPayloadParts > 1 && dh->splitPayloadParts != std::numeric_limits<decltype(dh->splitPayloadParts)>::max()) {
        //  if (lastSplitPartIndex == -1 && dh->splitPayloadIndex != 0) {
        //    LOG(warning) << "wrong split part index, expecting the first of " << dh->splitPayloadParts << " part(s)";
        //  } else if (dh->splitPayloadIndex != lastSplitPartIndex + 1) {
        //    LOG(warning) << "unordered split parts, expecting part " << lastSplitPartIndex + 1 << ", got " << dh->splitPayloadIndex
        //                 << " of " << dh->splitPayloadParts;
        //  } else if (channelNameForSplitParts.empty() == false && channelName != channelNameForSplitParts) {
        //    LOG(error) << "inconsistent channel for split part " << dh->splitPayloadIndex
        //               << ", matching " << channelName << ", expecting " << channelNameForSplitParts;
        //  }
        //}
        LOGP(debug, "associating {} part(s) at index {} to channel {} ({})", finalBlockIndex - msgidx, msgidx, channelName, outputs[channelName].Size());
        for (; msgidx < finalBlockIndex; ++msgidx) {
          outputs[channelName].AddPart(std::move(parts.At(msgidx)));
        }
        msgidx -= 2;
      } else {
        msgidx = finalBlockIndex - 2;
      }
      if (finalBlockIndex == 0 && !DataSpecUtils::match(query, "DPL", "EOS", 0)) {
        unmatchedDescriptions.emplace_back(DataSpecUtils::describe(query));
      }
    } // end of loop over parts

    for (auto& [channelName, channelParts] : outputs) {
      if (channelParts.Size() == 0) {
        continue;
      }
      sendOnChannel(device, channelParts, channelName, dplCounter);
    }
    if (not unmatchedDescriptions.empty()) {
      if (throwOnUnmatchedInputs) {
        std::string descriptions;
        for (auto const& desc : unmatchedDescriptions) {
          descriptions += "\n   " + desc;
        }
        throw std::runtime_error("No matching filter rule for input data " + descriptions +
                                 "\n Add appropriate matcher(s) to dataspec definition or allow to drop unmatched data");
      } else {
        bool changed = false;
        for (auto const& desc : unmatchedDescriptions) {
          if (not droppedDataSpecs->find(desc)) {
            // a new description
            droppedDataSpecs->add(desc);
            changed = true;
          }
        }
        if (changed) {
          droppedDataSpecs->warning();
        }
      }
    }
  };
}

InjectorFunction incrementalConverter(OutputSpec const& spec, uint64_t startTime, uint64_t step)
{
  auto timesliceId = std::make_shared<size_t>(startTime);

  return [timesliceId, spec, step](TimingInfo&, FairMQDevice& device, FairMQParts& parts, ChannelRetriever channelRetriever) {
    // We iterate on all the parts and we send them two by two,
    // adding the appropriate O2 header.
    for (int i = 0; i < parts.Size(); ++i) {
      DataHeader dh;

      // FIXME: this only supports fully specified output specs...
      ConcreteDataMatcher matcher = DataSpecUtils::asConcreteDataMatcher(spec);
      dh.dataOrigin = matcher.origin;
      dh.dataDescription = matcher.description;
      dh.subSpecification = matcher.subSpec;
      dh.payloadSize = parts.At(i)->GetSize();

      DataProcessingHeader dph{*timesliceId, 0};
      *timesliceId += step;
      // we have to move the incoming data
      o2::header::Stack headerStack{dh, dph};

      sendOnChannel(device, std::move(headerStack), std::move(parts.At(i)), spec, channelRetriever);
    }
  };
}

DataProcessorSpec specifyExternalFairMQDeviceProxy(char const* name,
                                                   std::vector<OutputSpec> const& outputs,
                                                   char const* defaultChannelConfig,
                                                   std::function<void(TimingInfo&,
                                                                      FairMQDevice&,
                                                                      FairMQParts&,
                                                                      ChannelRetriever)>
                                                     converter,
                                                   uint64_t minSHM)
{
  DataProcessorSpec spec;
  spec.name = strdup(name);
  spec.inputs = {};
  spec.outputs = outputs;
  // The Init method will register a new "Out of band" channel and
  // attach an OnData to it which is responsible for converting incoming
  // messages into DPL messages.
  spec.algorithm = AlgorithmSpec{[converter, channel = spec.name, minSHM](InitContext& ctx) {
    auto device = ctx.services().get<RawDeviceService>().device();
    // make a copy of the output routes and pass to the lambda by move
    auto outputRoutes = ctx.services().get<RawDeviceService>().spec().outputs;
    auto outputChannels = ctx.services().get<RawDeviceService>().spec().outputChannels;
    assert(device);

    // check that the name used for registering the OnData callback corresponds
    // to the configured output channel, unfortunately we can not automatically
    // deduce this from list of channels without knowing the name, because there
    // will be multiple channels. At least we throw a more informative exception.
    // FairMQDevice calls the custom init before the channels have been configured
    // so we do the check before starting in a dedicated callback
    auto channelConfigurationChecker = [channel, device]() {
      if (device->fChannels.count(channel) == 0) {
        throw std::runtime_error("the required out-of-band channel '" + channel + "' has not been configured, please check the name in the channel configuration");
      }
    };
    ctx.services().get<CallbackService>().set(CallbackService::Id::Start, channelConfigurationChecker);
    // Converter should pump messages

    auto dataHandler = [device, converter,
                        outputRoutes = std::move(outputRoutes),
                        control = &ctx.services().get<ControlService>(),
                        &timingInfo = ctx.services().get<TimingInfo>(),
                        outputChannels = std::move(outputChannels)](FairMQParts& inputs, int) {
      // pass a copy of the outputRoutes
      auto channelRetriever = [&outputRoutes](OutputSpec const& query, DataProcessingHeader::StartTime timeslice) -> std::string {
        for (auto& route : outputRoutes) {
          LOG(debug) << "matching: " << DataSpecUtils::describe(query) << " to route " << DataSpecUtils::describe(route.matcher);
          if (DataSpecUtils::match(route.matcher, query) && ((timeslice % route.maxTimeslices) == route.timeslice)) {
            return route.channel;
          }
        }
        return {""};
      };

      auto checkEos = [&inputs]() -> bool {
        std::string channelNameForSplitParts;
        for (int msgidx = 0; msgidx < inputs.Size() / 2; ++msgidx) {
          auto const sih = o2::header::get<SourceInfoHeader*>(inputs.At(msgidx * 2)->GetData());
          if (sih != nullptr && sih->state == InputChannelState::Completed) {
            return true;
          }
        }
        return false;
      };
      // we buffer the condition since the converter will forward messages by move
      bool doEos = checkEos();
      converter(timingInfo, *device, inputs, channelRetriever);

      if (doEos) {
        for (auto const& channel : outputChannels) {
          DataProcessingHelpers::sendEndOfStream(*device, channel);
        }
        control->endOfStream();
      }
    };

    auto runHandler = [dataHandler, channel, minSHM](ProcessingContext& ctx) {
      static RateLimiter limiter;
      auto device = ctx.services().get<RawDeviceService>().device();
      limiter.check(ctx, std::stoi(device->fConfig->GetValue<std::string>("timeframes-rate-limit")), minSHM);

      FairMQParts parts;
      device->Receive(parts, channel, 0);
      // Populate TimingInfo from the first message
      if (parts.Size() != 0) {
        auto const dh = o2::header::get<DataHeader*>(parts.At(0)->GetData());
        auto& timingInfo = ctx.services().get<TimingInfo>();
        if (dh != nullptr) {
          timingInfo.runNumber = dh->runNumber;
          timingInfo.firstTFOrbit = dh->firstTForbit;
          timingInfo.tfCounter = dh->tfCounter;
        }
        auto const dph = o2::header::get<DataProcessingHeader*>(parts.At(0)->GetData());
        if (dph != nullptr) {
          timingInfo.timeslice = dph->startTime;
          timingInfo.creation = dph->creation;
        }
      }
      dataHandler(parts, 0);
    };

    return runHandler;
  }};
  const char* d = strdup(((std::string(defaultChannelConfig).find("name=") == std::string::npos ? (std::string("name=") + name + ",") : "") + std::string(defaultChannelConfig)).c_str());
  spec.options = {
    ConfigParamSpec{"channel-config", VariantType::String, d, {"Out-of-band channel config"}}};
  return spec;
}

// Decide where to sent the output. Everything to "downstream" if there is such a channel.
std::string defaultOutputProxyChannelSelector(InputSpec const& input, const std::unordered_map<std::string, std::vector<FairMQChannel>>& channels)
{
  return channels.count("downstream") ? "downstream" : input.binding;
}

DataProcessorSpec specifyFairMQDeviceOutputProxy(char const* name,
                                                 Inputs const& inputSpecs,
                                                 const char* defaultChannelConfig)
{
  DataProcessorSpec spec;
  spec.name = name;
  spec.inputs = inputSpecs;
  spec.outputs = {};
  spec.algorithm = adaptStateful([inputSpecs](CallbackService& callbacks, RawDeviceService& rds, DeviceSpec const& deviceSpec, ConfigParamRegistry const& options) {
    // we can retrieve the channel name from the channel configuration string
    // FIXME: even if a --channel-config option is specified on the command line, always the default string
    // is retrieved from the config registry. The channel name thus needs to be configured in the default
    // string AND must match the name in an optional channel config.
    auto channelConfig = options.get<std::string>("channel-config");
    std::regex r{R"(name=([^,]*))"};
    std::vector<std::string> values{std::sregex_token_iterator{std::begin(channelConfig), std::end(channelConfig), r, 1},
                                    std::sregex_token_iterator{}};
    if (values.size() != 1 || values[0].empty()) {
      throw std::runtime_error("failed to extract channel name from channel configuration parameter '" + channelConfig + "'");
    }
    std::string outputChannelName = values[0];

    auto* device = rds.device();
    // check that the input spec bindings have corresponding output channels
    // FairMQDevice calls the custom init before the channels have been configured
    // so we do the check before starting in a dedicated callback
    auto channelConfigurationChecker = [inputSpecs = std::move(inputSpecs), device, outputChannelName]() {
      LOG(info) << "checking channel configuration";
      if (device->fChannels.count(outputChannelName) == 0) {
        throw std::runtime_error("no corresponding output channel found for input '" + outputChannelName + "'");
      }
    };
    callbacks.set(CallbackService::Id::Start, channelConfigurationChecker);
    auto lastDataProcessingHeader = std::make_shared<DataProcessingHeader>(0, 0);

    if (deviceSpec.forwards.size() > 0) {
      // check that no internal forwards are existing, i.e. that proxy is at the end of the workflow
      // in principle we can be less strict here if we check only for the defined input specs that there
      // are no internal forwards
      throw std::runtime_error("can not add forward targets outside DPL if internal forwards are existing, the proxy must be at the end of the workflow");
    }
    for (auto const& inputSpec : inputSpecs) {
      // this is a prototype, in principle we want to have all spec objects const
      // and so only the const object can be retrieved from service registry
      ForwardRoute route{0, 1, inputSpec, outputChannelName};
      const_cast<DeviceSpec&>(deviceSpec).forwards.emplace_back(route);
    }

    auto forwardEos = [device, lastDataProcessingHeader, outputChannelName](EndOfStreamContext&) {
      // DPL implements an internal end of stream signal, which is propagated through
      // all downstream channels if a source is dry, make it available to other external
      // devices via a message of type {DPL/EOS/0}
      for (auto& channelInfo : device->fChannels) {
        auto& channelName = channelInfo.first;
        if (channelName != outputChannelName) {
          continue;
        }
        DataHeader dh;
        dh.dataOrigin = "DPL";
        dh.dataDescription = "EOS";
        dh.subSpecification = 0;
        dh.payloadSize = 0;
        dh.payloadSerializationMethod = o2::header::gSerializationMethodNone;
        dh.tfCounter = 0;
        dh.firstTForbit = 0;
        SourceInfoHeader sih;
        sih.state = InputChannelState::Completed;
        // allocate the header message using the underlying transport of the channel
        auto channelAlloc = o2::pmr::getTransportAllocator(channelInfo.second[0].Transport());
        auto headerMessage = o2::pmr::getMessage(o2::header::Stack{channelAlloc, dh, *lastDataProcessingHeader, sih});
        FairMQParts out;
        out.AddPart(std::move(headerMessage));
        // add empty payload message
        out.AddPart(device->NewMessageFor(channelName, 0, 0));
        sendOnChannel(*device, out, channelName, (size_t)-1);
      }
    };
    callbacks.set(CallbackService::Id::EndOfStream, forwardEos);

    return adaptStateless([lastDataProcessingHeader](InputRecord& inputs) {
      for (size_t ii = 0; ii != inputs.size(); ++ii) {
        for (size_t pi = 0; pi < inputs.getNofParts(ii); ++pi) {
          auto part = inputs.getByPos(ii, pi);
          const auto* dph = o2::header::get<DataProcessingHeader*>(part.header);
          if (dph) {
            // FIXME: should we implement an assignment operator for DataProcessingHeader?
            lastDataProcessingHeader->startTime = dph->startTime;
            lastDataProcessingHeader->duration = dph->duration;
            lastDataProcessingHeader->creation = dph->creation;
          }
        }
      }
    });
  });
  const char* d = strdup(((std::string(defaultChannelConfig).find("name=") == std::string::npos ? (std::string("name=") + name + ",") : "") + std::string(defaultChannelConfig)).c_str());
  spec.options = {
    ConfigParamSpec{"channel-config", VariantType::String, d, {"Out-of-band channel config"}},
  };

  return spec;
}

DataProcessorSpec specifyFairMQDeviceMultiOutputProxy(char const* name,
                                                      Inputs const& inputSpecs,
                                                      const char* defaultChannelConfig,
                                                      ChannelSelector channelSelector)
{
  // FIXME: this looks like a code duplication with the function above, check if the
  // two can be combined
  DataProcessorSpec spec;
  spec.name = name;
  spec.inputs = inputSpecs;
  spec.outputs = {};
  spec.algorithm = adaptStateful([inputSpecs, channelSelector](CallbackService& callbacks, RawDeviceService& rds, const DeviceSpec& deviceSpec) {
    auto device = rds.device();
    // check that the input spec bindings have corresponding output channels
    // FairMQDevice calls the custom init before the channels have been configured
    // so we do the check before starting in a dedicated callback
    // also we set forwards for all input specs and keep a list of all channels so we can send EOS on them
    auto channelNames = std::make_shared<std::vector<std::string>>();
    auto channelConfigurationInitializer = [inputSpecs = std::move(inputSpecs), device, channelSelector, &deviceSpec, channelNames]() {
      if (deviceSpec.forwards.size() > 0) {
        // check that no internal forwards are existing, i.e. that proxy is at the end of the workflow
        // in principle we can be less strict here if we check only for the defined input specs that there
        // are no internal forwards
        throw std::runtime_error("can not add forward targets outside DPL if internal forwards are existing, the proxy must be at the end of the workflow");
      }
      for (auto const& spec : inputSpecs) {
        auto channel = channelSelector(spec, device->fChannels);
        if (device->fChannels.count(channel) == 0) {
          throw std::runtime_error("no corresponding output channel found for input '" + channel + "'");
        }
        ForwardRoute route{0, 1, spec, channel};
        // this we will try to fix on the framework level, there will be an API to
        // set external routes. Basically, this has to be added while setting up the
        // workflow. After that, the actual spec provided by the service is supposed
        // to be const by design
        const_cast<DeviceSpec&>(deviceSpec).forwards.emplace_back(route);

        channelNames->emplace_back(std::move(channel));
      }
    };
    callbacks.set(CallbackService::Id::Start, channelConfigurationInitializer);

    auto lastDataProcessingHeader = std::make_shared<DataProcessingHeader>(0, 0);
    auto forwardEos = [device, lastDataProcessingHeader, channelNames](EndOfStreamContext&) {
      // DPL implements an internal end of stream signal, which is propagated through
      // all downstream channels if a source is dry, make it available to other external
      // devices via a message of type {DPL/EOS/0}
      for (auto& channelInfo : device->fChannels) {
        auto& channelName = channelInfo.first;
        auto checkChannel = [channelNames = std::move(*channelNames)](std::string const& name) -> bool {
          for (auto const& n : channelNames) {
            if (n == name) {
              return true;
            }
          }
          return false;
        };
        if (!checkChannel(channelName)) {
          continue;
        }
        DataHeader dh;
        dh.dataOrigin = "DPL";
        dh.dataDescription = "EOS";
        dh.subSpecification = 0;
        dh.payloadSize = 0;
        dh.payloadSerializationMethod = o2::header::gSerializationMethodNone;
        dh.tfCounter = 0;
        dh.firstTForbit = 0;
        SourceInfoHeader sih;
        sih.state = InputChannelState::Completed;
        // allocate the header message using the underlying transport of the channel
        auto channelAlloc = o2::pmr::getTransportAllocator(channelInfo.second[0].Transport());
        auto headerMessage = o2::pmr::getMessage(o2::header::Stack{channelAlloc, dh, *lastDataProcessingHeader, sih});
        FairMQParts out;
        out.AddPart(std::move(headerMessage));
        // add empty payload message
        out.AddPart(device->NewMessageFor(channelName, 0, 0));
        sendOnChannel(*device, out, channelName, (size_t)-1);
      }
    };
    callbacks.set(CallbackService::Id::EndOfStream, forwardEos);

    return adaptStateless([channelSelector, lastDataProcessingHeader](InputRecord& inputs) {
      // there is nothing to do if the forwarding is handled on the framework level
      // as forward routes but we need to keep a copy of the last DataProcessingHeader
      // for sending the EOS
      for (size_t ii = 0; ii != inputs.size(); ++ii) {
        for (size_t pi = 0; pi < inputs.getNofParts(ii); ++pi) {
          auto part = inputs.getByPos(ii, pi);
          const auto* dph = o2::header::get<DataProcessingHeader*>(part.header);
          if (dph) {
            // FIXME: should we implement an assignment operator for DataProcessingHeader?
            lastDataProcessingHeader->startTime = dph->startTime;
            lastDataProcessingHeader->duration = dph->duration;
            lastDataProcessingHeader->creation = dph->creation;
          }
        }
      }
    });
  });
  const char* d = strdup(((std::string(defaultChannelConfig).find("name=") == std::string::npos ? (std::string("name=") + name + ",") : "") + std::string(defaultChannelConfig)).c_str());
  spec.options = {
    ConfigParamSpec{"channel-config", VariantType::String, d, {"Out-of-band channel config"}},
  };

  return spec;
}

} // namespace o2::framework
