// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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
#include "Headers/DataHeader.h"
#include "Headers/Stack.h"

#include "./DeviceSpecHelpers.h"

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

void sendOnChannel(FairMQDevice& device, FairMQParts& messages, std::string const& channel)
{
  // Note: DPL is only setting up one instance of a channel while FairMQ allows to have an
  // array of channels, the index is 0 in the call
  constexpr auto index = 0;
  LOG(DEBUG) << "sending " << messages.Size() << " messages on " << channel;
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
      LOG(ERROR) << "failed to dispatch messages on channel " << channel << ", downstream queue might be full\n"
                 << "or unconnected. No data is dropped, keep on trying, but this will hold the reading from\n"
                 << "the input and expose back-pressure upstream. RESOLVE DOWNSTREAM CONGESTION to continue";
      if (timeout == maxTimeout) {
        // we add 1ms to disable the warning below
        timeout += 1;
      }
    }
    if (device.NewStatePending()) {
      LOG(ERROR) << "device state change is requested, dropping " << messages.Size() << " pending message(s)\n"
                 << "on channel " << channel << "\n"
                 << "ATTENTION: DATA IS LOST! Could not dispatch data to downstream consumer(s), check if\n"
                 << "consumers have been terminated too early";
      // make sure we disable the warning below
      timeout = maxTimeout + 1;
      break;
    }
  }
  if (timeout > 0 && timeout <= maxTimeout) {
    LOG(WARNING) << "dispatching on channel " << channel << " was delayed by " << timeout << " ms";
  }
  // TODO: feeling this is a bit awkward, but the interface of FairMQParts does not provide a
  // method to clear the content.
  // Maybe the FairMQ API can be improved at some point. Actually the ownership of all messages should be passed
  // on to the transport and the messages should be empty after sending and the parts content can be cleared.
  //assert(std::accumulate(messages.begin(), messages.end(), true, [](bool a, auto const& msg) {return a && (msg.get() == nullptr);}));
  messages.fParts.clear();
}

void sendOnChannel(FairMQDevice& device, FairMQParts& messages, OutputSpec const& spec, DataProcessingHeader::StartTime tslice, ChannelRetriever& channelRetriever)
{
  // Note: DPL is only setting up one instance of a channel while FairMQ allows to have an
  // array of channels, the index is 0 in the call
  auto channel = channelRetriever(spec, tslice);
  if (channel.empty()) {
    LOG(WARNING) << "can not find matching channel for " << DataSpecUtils::describe(spec) << " timeslice " << tslice;
    return;
  }
  sendOnChannel(device, messages, channel);
}

void sendOnChannel(FairMQDevice& device, o2::header::Stack&& headerStack, FairMQMessagePtr&& payloadMessage, OutputSpec const& spec, ChannelRetriever& channelRetriever)
{
  const auto* dph = o2::header::get<DataProcessingHeader*>(headerStack.data());
  if (!dph) {
    LOG(ERROR) << "Header Stack does not follow the O2 data model, DataProcessingHeader missing";
    return;
  }
  auto channelName = channelRetriever(spec, dph->startTime);
  constexpr auto index = 0;
  if (channelName.empty()) {
    LOG(WARNING) << "can not find matching channel for " << DataSpecUtils::describe(spec);
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
    sendOnChannel(device, out, channelName);
    return;
  }
  LOG(ERROR) << "internal mismatch, can not find channel " << channelName << " in the list of channel infos of the device";
}

void sendOnChannel(FairMQDevice& device, FairMQMessagePtr&& headerMessage, FairMQMessagePtr&& payloadMessage, OutputSpec const& spec, ChannelRetriever& channelRetriever)
{
  //  const auto* dph = o2::header::get<DataProcessingHeader*>( *reinterpret_cast<o2::header::Stack*>(headerMessage->GetData()) );
  const auto* dph = o2::header::get<DataProcessingHeader*>(headerMessage->GetData());
  if (!dph) {
    LOG(ERROR) << "Header does not follow the O2 data model, DataProcessingHeader missing";
    return;
  }
  auto tslice = dph->startTime;
  FairMQParts out;
  out.AddPart(std::move(headerMessage));
  out.AddPart(std::move(payloadMessage));
  sendOnChannel(device, out, spec, tslice, channelRetriever);
}

InjectorFunction o2DataModelAdaptor(OutputSpec const& spec, uint64_t startTime, uint64_t step)
{
  auto timesliceId = std::make_shared<size_t>(startTime);
  return [timesliceId, step, spec](FairMQDevice& device, FairMQParts& parts, ChannelRetriever channelRetriever) {
    for (size_t i = 0; i < parts.Size() / 2; ++i) {
      auto dh = o2::header::get<DataHeader*>(parts.At(i * 2)->GetData());

      DataProcessingHeader dph{*timesliceId, 0};
      o2::header::Stack headerStack{*dh, dph};
      sendOnChannel(device, std::move(headerStack), std::move(parts.At(i * 2 + 1)), spec, channelRetriever);
      auto oldTimesliceId = *timesliceId;
      *timesliceId += 1;
    }
  };
}

InjectorFunction dplModelAdaptor(std::vector<OutputSpec> const& filterSpecs, bool throwOnUnmatchedInputs)
{
  // structure to hald information on the unmatch ed data and print a warning at cleanup
  class DroppedDataSpecs
  {
   public:
    DroppedDataSpecs() = default;
    ~DroppedDataSpecs()
    {
      warning();
    }

    bool find(std::string const& desc) const
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
        LOG(WARNING) << "Some input data are not matched by filter rules " << descriptions << "\n"
                     << "DROPPING OF THESE MESSAGES HAS BEEN ENABLED BY CONFIGURATION";
      }
    }

   private:
    std::string descriptions;
  };

  return [filterSpecs = std::move(filterSpecs), throwOnUnmatchedInputs, droppedDataSpecs = std::make_shared<DroppedDataSpecs>()](FairMQDevice& device, FairMQParts& parts, ChannelRetriever channelRetriever) {
    std::unordered_map<std::string, FairMQParts> outputs;
    std::vector<bool> indicesDone(parts.Size() / 2, false);
    std::vector<std::string> unmatchedDescriptions;
    int lastSplitPartIndex = -1;
    std::string channelNameForSplitParts;
    for (size_t msgidx = 0; msgidx < parts.Size() / 2; ++msgidx) {
      if (indicesDone[msgidx]) {
        continue;
      }

      const auto dh = o2::header::get<DataHeader*>(parts.At(msgidx * 2)->GetData());
      if (!dh) {
        LOG(ERROR) << "data on input " << msgidx << " does not follow the O2 data model, DataHeader missing";
        continue;
      }
      const auto dph = o2::header::get<DataProcessingHeader*>(parts.At(msgidx * 2)->GetData());
      if (!dph) {
        LOG(ERROR) << "data on input " << msgidx << " does not follow the O2 data model, DataProcessingHeader missing";
        continue;
      }
      LOG(DEBUG) << msgidx << ": " << DataSpecUtils::describe(OutputSpec{dh->dataOrigin, dh->dataDescription, dh->subSpecification}) << " part " << dh->splitPayloadIndex << " of " << dh->splitPayloadParts << "  payload " << parts.At(msgidx * 2 + 1)->GetSize();

      OutputSpec query{dh->dataOrigin, dh->dataDescription, dh->subSpecification};
      LOG(DEBUG) << "processing " << DataSpecUtils::describe(OutputSpec{dh->dataOrigin, dh->dataDescription, dh->subSpecification}) << " time slice " << dph->startTime << " part " << dh->splitPayloadIndex << " of " << dh->splitPayloadParts;
      for (auto const& spec : filterSpecs) {
        // filter on the specified OutputSpecs, the default value is a ConcreteDataTypeMatcher with origin and description 'any'
        if (DataSpecUtils::match(spec, OutputSpec{{header::gDataOriginAny, header::gDataDescriptionAny}}) ||
            DataSpecUtils::match(spec, query)) {
          auto channelName = channelRetriever(query, dph->startTime);
          if (channelName.empty()) {
            LOG(WARNING) << "can not find matching channel, not able to adopt " << DataSpecUtils::describe(query);
            break;
          }
          // the checks for consistency of split payload parts are of informative nature
          // forwarding happens independently
          if (dh->splitPayloadParts > 1 && dh->splitPayloadParts != std::numeric_limits<decltype(dh->splitPayloadParts)>::max()) {
            if (lastSplitPartIndex == -1 && dh->splitPayloadIndex != 0) {
              LOG(WARNING) << "wrong split part index, expecting the first of " << dh->splitPayloadParts << " part(s)";
            } else if (dh->splitPayloadIndex != lastSplitPartIndex + 1) {
              LOG(WARNING) << "unordered split parts, expecting part " << lastSplitPartIndex + 1 << ", got " << dh->splitPayloadIndex
                           << " of " << dh->splitPayloadParts;
            } else if (channelNameForSplitParts.empty() == false && channelName != channelNameForSplitParts) {
              LOG(ERROR) << "inconsistent channel for split part " << dh->splitPayloadIndex
                         << ", matching " << channelName << ", expecting " << channelNameForSplitParts;
            }
            lastSplitPartIndex = dh->splitPayloadIndex;
            channelNameForSplitParts = channelName;
            if (lastSplitPartIndex + 1 == dh->splitPayloadParts) {
              lastSplitPartIndex = -1;
              channelNameForSplitParts = "";
            }
          } else if (lastSplitPartIndex != -1) {
            LOG(WARNING) << "found incomplete or unordered split parts, expecting part " << lastSplitPartIndex + 1
                         << " but got a new data block";
          }
          outputs[channelName].AddPart(std::move(parts.At(msgidx * 2)));
          outputs[channelName].AddPart(std::move(parts.At(msgidx * 2 + 1)));
          LOG(DEBUG) << "associating part with index " << msgidx << " to channel " << channelName << " (" << outputs[channelName].Size() << ")";
          indicesDone[msgidx] = true;
          break;
        }
      }
      if (indicesDone[msgidx] == false) {
        unmatchedDescriptions.emplace_back(DataSpecUtils::describe(query));
      }
    }
    for (auto& [channelName, channelParts] : outputs) {
      if (channelParts.Size() == 0) {
        continue;
      }
      sendOnChannel(device, channelParts, channelName);
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

  return [timesliceId, spec, step](FairMQDevice& device, FairMQParts& parts, ChannelRetriever channelRetriever) {
    // We iterate on all the parts and we send them two by two,
    // adding the appropriate O2 header.
    for (size_t i = 0; i < parts.Size(); ++i) {
      DataHeader dh;

      // FIXME: this only supports fully specified output specs...
      ConcreteDataMatcher matcher = DataSpecUtils::asConcreteDataMatcher(spec);
      dh.dataOrigin = matcher.origin;
      dh.dataDescription = matcher.description;
      dh.subSpecification = matcher.subSpec;
      dh.payloadSize = parts.At(i)->GetSize();

      DataProcessingHeader dph{*timesliceId, 0};
      *timesliceId += step;
      //we have to move the incoming data
      o2::header::Stack headerStack{dh, dph};

      sendOnChannel(device, std::move(headerStack), std::move(parts.At(i)), spec, channelRetriever);
    }
  };
}

DataProcessorSpec specifyExternalFairMQDeviceProxy(char const* name,
                                                   std::vector<OutputSpec> const& outputs,
                                                   char const* defaultChannelConfig,
                                                   std::function<void(FairMQDevice&,
                                                                      FairMQParts&,
                                                                      ChannelRetriever)>
                                                     converter)
{
  DataProcessorSpec spec;
  spec.name = strdup(name);
  spec.inputs = {};
  spec.outputs = outputs;
  // The Init method will register a new "Out of band" channel and
  // attach an OnData to it which is responsible for converting incoming
  // messages into DPL messages.
  spec.algorithm = AlgorithmSpec{[converter, channel = spec.name](InitContext& ctx) {
    auto device = ctx.services().get<RawDeviceService>().device();
    // make a copy of the output routes and pass to the lambda by move
    auto outputRoutes = ctx.services().get<RawDeviceService>().spec().outputs;
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
    auto handler = [device, converter, outputRoutes = std::move(outputRoutes)](FairMQParts& inputs, int) {
      auto channelRetriever = [outputRoutes = std::move(outputRoutes)](OutputSpec const& query, DataProcessingHeader::StartTime timeslice) -> std::string {
        for (auto& route : outputRoutes) {
          LOG(DEBUG) << "matching: " << DataSpecUtils::describe(query) << " to route " << DataSpecUtils::describe(route.matcher);
          if (DataSpecUtils::match(route.matcher, query) && ((timeslice % route.maxTimeslices) == route.timeslice)) {
            return route.channel;
          }
        }
        return std::string("");
      };
      converter(*device, inputs, channelRetriever);
      return true;
    };

    device->OnData(channel, handler);
    return [](ProcessingContext&) {};
  }};
  const char* d = strdup(((std::string(defaultChannelConfig).find("name=") == std::string::npos ? (std::string("name=") + name + ",") : "") + std::string(defaultChannelConfig)).c_str());
  spec.options = {
    ConfigParamSpec{"channel-config", VariantType::String, d, {"Out-of-band channel config"}}};
  return spec;
}

namespace
{
static char const* gDefaultChannel = "downstream";
// Decide where to sent the output. Everything to "downstream" if there is such a channel.
std::string decideChannel(InputSpec const& input, const std::unordered_map<std::string, std::vector<FairMQChannel>>& channels)
{
  return channels.count("downstream") ? "downstream" : input.binding;
}
} // namespace

DataProcessorSpec specifyFairMQDeviceOutputProxy(char const* name,
                                                 Inputs const& inputSpecs,
                                                 const char* defaultChannelConfig)
{
  DataProcessorSpec spec;
  spec.name = name;
  spec.inputs = inputSpecs;
  spec.outputs = {};
  spec.algorithm = adaptStateful([inputSpecs](CallbackService& callbacks, RawDeviceService& rds) {
    auto device = rds.device();
    // check that the input spec bindings have corresponding output channels
    // FairMQDevice calls the custom init before the channels have been configured
    // so we do the check before starting in a dedicated callback
    auto channelConfigurationChecker = [inputSpecs = std::move(inputSpecs), device]() {
      LOG(INFO) << "checking channel configuration";
      if (device->fChannels.count("downstream") == 0) {
        throw std::runtime_error("no corresponding output channel found for input 'downstream'");
      }
    };
    callbacks.set(CallbackService::Id::Start, channelConfigurationChecker);
    return adaptStateless([](RawDeviceService& rds, InputRecord& inputs) {
      FairMQParts outputs;
      auto& device = *rds.device();
      for (size_t ii = 0; ii != inputs.size(); ++ii) {
        auto first = inputs.getByPos(ii, 0);
        // we could probably do something like this but we do not know when the message is going to be sent
        // and if DPL is still owning a valid copy.
        //auto headerMessage = device.NewMessageFor(input.spec->binding, input.header, headerMsgSize, [](void*, void*) {});

        // Note: DPL is only setting up one instance of a channel while FairMQ allows to have an
        // array of channels, the index is 0 in the call
        constexpr auto index = 0;
        for (size_t pi = 0; pi < inputs.getNofParts(ii); ++pi) {
          auto part = inputs.getByPos(ii, pi);
          // TODO: we need to make a copy of the messages, maybe we can implement functionality in
          // the RawDeviceService to forward messages, but this also needs to take into account that
          // other consumers might exist
          size_t headerMsgSize = o2::header::Stack::headerStackSize(reinterpret_cast<o2::byte const*>(part.header));
          auto* dh = o2::header::get<DataHeader*>(part.header);
          if (!dh) {
            std::stringstream errorMessage;
            errorMessage << "no data header in " << *first.spec;
            throw std::runtime_error(errorMessage.str());
          }
          size_t payloadMsgSize = dh->payloadSize;

          // FIXME: there is a copy here. Why?????
          auto headerMessage = device.NewMessageFor("downstream", index, headerMsgSize);
          memcpy(headerMessage->GetData(), part.header, headerMsgSize);
          auto payloadMessage = device.NewMessageFor("downstream", index, payloadMsgSize);
          memcpy(payloadMessage->GetData(), part.payload, payloadMsgSize);
          outputs.AddPart(std::move(headerMessage));
          outputs.AddPart(std::move(payloadMessage));
        }
      }
      sendOnChannel(device, outputs, "downstream");
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
                                                      const char* defaultChannelConfig)
{
  DataProcessorSpec spec;
  spec.name = name;
  spec.inputs = inputSpecs;
  spec.outputs = {};
  spec.algorithm = adaptStateful([inputSpecs](CallbackService& callbacks, RawDeviceService& rds) {
    auto device = rds.device();
    // check that the input spec bindings have corresponding output channels
    // FairMQDevice calls the custom init before the channels have been configured
    // so we do the check before starting in a dedicated callback
    auto channelConfigurationChecker = [inputSpecs = std::move(inputSpecs), device]() {
      LOG(INFO) << "checking channel configuration";
      for (auto const& spec : inputSpecs) {
        auto channel = decideChannel(spec, device->fChannels);
        if (device->fChannels.count(channel) == 0) {
          throw std::runtime_error("no corresponding output channel found for input '" + channel + "'");
        }
      }
    };
    callbacks.set(CallbackService::Id::Start, channelConfigurationChecker);
    return adaptStateless([](RawDeviceService& rds, InputRecord& inputs) {
      std::unordered_map<std::string, FairMQParts> outputs;
      auto& device = *rds.device();
      for (size_t ii = 0; ii != inputs.size(); ++ii) {
        auto first = inputs.getByPos(ii, 0);
        // we could probably do something like this but we do not know when the message is going to be sent
        // and if DPL is still owning a valid copy.
        //auto headerMessage = device.NewMessageFor(input.spec->binding, input.header, headerMsgSize, [](void*, void*) {});

        // Note: DPL is only setting up one instance of a channel while FairMQ allows to have an
        // array of channels, the index is 0 in the call
        constexpr auto index = 0;
        for (size_t pi = 0; pi < inputs.getNofParts(ii); ++pi) {
          auto part = inputs.getByPos(ii, pi);
          // TODO: we need to make a copy of the messages, maybe we can implement functionality in
          // the RawDeviceService to forward messages, but this also needs to take into account that
          // other consumers might exist
          size_t headerMsgSize = o2::header::Stack::headerStackSize(reinterpret_cast<o2::byte const*>(part.header));
          auto* dh = o2::header::get<DataHeader*>(part.header);
          if (!dh) {
            std::stringstream errorMessage;
            errorMessage << "no data header in " << *first.spec;
            throw std::runtime_error(errorMessage.str());
          }
          size_t payloadMsgSize = dh->payloadSize;

          auto channel = decideChannel(*first.spec, device.fChannels);
          auto headerMessage = device.NewMessageFor(channel, index, headerMsgSize);
          memcpy(headerMessage->GetData(), part.header, headerMsgSize);
          auto payloadMessage = device.NewMessageFor(channel, index, payloadMsgSize);
          memcpy(payloadMessage->GetData(), part.payload, payloadMsgSize);
          outputs[channel].AddPart(std::move(headerMessage));
          outputs[channel].AddPart(std::move(payloadMessage));
        }
      }
      for (auto& [channelName, channelParts] : outputs) {
        if (channelParts.Size() == 0) {
          continue;
        }
        sendOnChannel(device, channelParts, channelName);
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
