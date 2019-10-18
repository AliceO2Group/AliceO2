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
#include "Framework/DataProcessingHeader.h"
#include "Framework/DataSpecUtils.h"
#include "Framework/DeviceSpec.h"
#include "Framework/ExternalFairMQDeviceProxy.h"
#include "Framework/InitContext.h"
#include "Framework/ProcessingContext.h"
#include "Framework/RawDeviceService.h"

#include "Headers/DataHeader.h"

#include <fairmq/FairMQParts.h>
#include <fairmq/FairMQDevice.h>
#include <cstring>
#include <cassert>
#include <memory>
#include <optional>
#include <unordered_map>

namespace o2
{
namespace framework
{

using DataHeader = o2::header::DataHeader;

void sendOnChannel(FairMQDevice& device, FairMQParts& messages, std::string const& channel)
{
  // Note: DPL is only setting up one instance of a channel while FairMQ allows to have an
  // array of channels, the index is 0 in the call
  constexpr auto index = 0;
  LOG(DEBUG) << "sending " << messages.Size() << " messages on " << channel;
  device.Send(messages, channel, index);
}

void sendOnChannel(FairMQDevice& device, FairMQParts& messages, OutputSpec const& spec, ChannelRetriever& channelRetriever)
{
  // Note: DPL is only setting up one instance of a channel while FairMQ allows to have an
  // array of channels, the index is 0 in the call
  auto channel = channelRetriever(spec);
  if (channel.empty()) {
    LOG(WARNING) << "can not find matching channel for " << DataSpecUtils::describe(spec);
    return;
  }
  sendOnChannel(device, messages, channel);
}

void sendOnChannel(FairMQDevice& device, o2::header::Stack&& headerStack, FairMQMessagePtr&& payloadMessage, OutputSpec const& spec, ChannelRetriever& channelRetriever)
{
  auto channelName = channelRetriever(spec);
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
    device.Send(out, channelName, 0);
    return;
  }
  LOG(ERROR) << "internal mismatch, can not find channel " << channelName << " in the list of channel infos of the device";
}

void sendOnChannel(FairMQDevice& device, FairMQMessagePtr&& headerMessage, FairMQMessagePtr&& payloadMessage, OutputSpec const& spec, ChannelRetriever& channelRetriever)
{
  FairMQParts out;
  out.AddPart(std::move(headerMessage));
  out.AddPart(std::move(payloadMessage));
  sendOnChannel(device, out, spec, channelRetriever);
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

std::tuple<std::vector<size_t>, size_t> findSplitParts(FairMQParts& parts, size_t start, std::vector<bool>& indicesDone)
{
  std::optional<ConcreteDataMatcher> matcher = std::nullopt;
  size_t nofParts = 0;
  bool isGood = true;
  std::vector<size_t> indexList;
  size_t payloadSize = 0;
  size_t index = start;
  do {
    auto dh = o2::header::get<DataHeader*>(parts.At(index * 2)->GetData());
    if (!dh) {
      LOG(ERROR) << "data on input " << index << " does not follow the O2 data model, DataHeader missing";
      break;
    }

    if (matcher == std::nullopt) {
      matcher = ConcreteDataMatcher{dh->dataOrigin, dh->dataDescription, dh->subSpecification};
      nofParts = dh->splitPayloadParts;
    } else if (*matcher != ConcreteDataMatcher{dh->dataOrigin, dh->dataDescription, dh->subSpecification}) {
      continue;
    }
    LOG(DEBUG) << "matched " << DataSpecUtils::describe(OutputSpec{dh->dataOrigin, dh->dataDescription, dh->subSpecification}) << " part " << dh->splitPayloadIndex << " of " << dh->splitPayloadParts;
    indicesDone[index] = true;

    if (isGood && (dh->splitPayloadIndex != indexList.size())) {
      LOG(ERROR) << "invalid sequence of split payload parts, expecting index " << indexList.size() << " but got " << dh->splitPayloadIndex;
      isGood = false;
    } else if (isGood && (dh->payloadSize != parts.At(index * 2 + 1)->GetSize())) {
      LOG(ERROR) << "Mismatch of payload size in DataHeader and size of payload message: " << dh->payloadSize << "/"
                 << parts.At(index * 2 + 1)->GetSize();
      isGood = false;
    }
    if (!isGood) {
      continue;
    }
    indexList.emplace_back(index);
    payloadSize += dh->payloadSize;
    LOG(DEBUG) << "processed index " << index << ", cached " << indexList.size() << ", total payload size " << payloadSize;
  } while ((++index) < parts.Size() / 2);

  if (indexList.size() != nofParts || !isGood) {
    payloadSize = 0;
    indexList.clear();
  }
  return std::make_tuple(std::move(indexList), payloadSize);
}

FairMQMessagePtr mergePayloads(FairMQDevice& device, FairMQParts& parts, std::vector<size_t> const& indexList, size_t payloadSize, std::string const channel)
{
  auto message = device.NewMessageFor(channel, 0, payloadSize);
  if (!message) {
    return message;
  }
  size_t copied = 0;
  char* data = reinterpret_cast<char*>(message->GetData());
  for (auto const& index : indexList) {
    size_t partSize = parts.At(index * 2 + 1)->GetSize();
    memcpy(data, parts.At(index * 2 + 1)->GetData(), partSize);
    data += partSize;
    copied += partSize;
  }
  return message;
}

InjectorFunction dplModelAdaptor(std::vector<OutputSpec> const& filterSpecs, bool throwOnUnmatchedInputs)
{
  return [filterSpecs = std::move(filterSpecs), throwOnUnmatchedInputs](FairMQDevice& device, FairMQParts& parts, ChannelRetriever channelRetriever) {
    std::unordered_map<std::string, FairMQParts> outputs;
    std::vector<bool> indicesDone(parts.Size() / 2, false);
    for (size_t i = 0; i < parts.Size() / 2; ++i) {
      if (indicesDone[i]) {
        continue;
      }
      auto dh = o2::header::get<DataHeader*>(parts.At(i * 2)->GetData());
      if (!dh) {
        LOG(ERROR) << "data on input " << i << " does not follow the O2 data model, DataHeader missing";
        continue;
      }
      OutputSpec query{dh->dataOrigin, dh->dataDescription, dh->subSpecification};
      LOG(DEBUG) << "processing " << DataSpecUtils::describe(OutputSpec{dh->dataOrigin, dh->dataDescription, dh->subSpecification}) << " part " << dh->splitPayloadIndex << " of " << dh->splitPayloadParts;
      for (auto const& spec : filterSpecs) {
        // filter on the specified OutputSpecs, the default value is a ConcreteDataTypeMatcher with origin and description 'any'
        if (DataSpecUtils::match(spec, OutputSpec{{header::gDataOriginAny, header::gDataDescriptionAny}}) ||
            DataSpecUtils::match(spec, query)) {
          auto channelName = channelRetriever(query);
          if (channelName.empty()) {
            LOG(WARNING) << "can not find matching channel, not able to adopt " << DataSpecUtils::describe(query);
            break;
          }
          if (dh->splitPayloadParts > 1 && dh->splitPayloadParts != std::numeric_limits<decltype(dh->splitPayloadParts)>::max()) {
            auto [indexList, payloadSize] = findSplitParts(parts, i, indicesDone);
            if (payloadSize > 0) {
              size_t headerSize = parts.At(i * 2)->GetSize();
              auto headerMessage = device.NewMessageFor(channelName, 0, headerSize);
              memcpy(headerMessage->GetData(), parts.At(i * 2)->GetData(), headerSize);
              auto clonedh = const_cast<DataHeader*>(o2::header::get<DataHeader*>(headerMessage->GetData()));
              clonedh->splitPayloadParts = 0;
              clonedh->splitPayloadIndex = 0;
              clonedh->payloadSize = payloadSize;
              outputs[channelName].AddPart(std::move(headerMessage));
              outputs[channelName].AddPart(std::move(mergePayloads(device, parts, indexList, payloadSize, channelName)));
              LOG(DEBUG) << "merged " << indexList.size() << " split payload parts in new message of size " << payloadSize << " on channel " << channelName << " (" << outputs[channelName].Size() << " parts)";
            }
          } else {
            outputs[channelName].AddPart(std::move(parts.At(i * 2)));
            outputs[channelName].AddPart(std::move(parts.At(i * 2 + 1)));
            LOG(DEBUG) << "associating part with index " << i << " to channel " << channelName << " (" << outputs[channelName].Size() << ")";
          }
          indicesDone[i] = true;
          break;
        } else if (throwOnUnmatchedInputs) {
          throw std::runtime_error("no matching filter rule for input data " + DataSpecUtils::describe(query));
        }
      }
    }
    for (auto& [channelName, channelParts] : outputs) {
      sendOnChannel(device, channelParts, channelName);
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
                                                   char const* channelConfig,
                                                   std::function<void(FairMQDevice&,
                                                                      FairMQParts&,
                                                                      std::function<std::string(OutputSpec const&)>)>
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

    // Converter should pump messages
    auto handler = [device, converter, outputRoutes = std::move(outputRoutes)](FairMQParts& inputs, int) {
      auto channelRetriever = [outputRoutes = std::move(outputRoutes)](OutputSpec const& query) -> std::string {
        for (auto& route : outputRoutes) {
          LOG(DEBUG) << "matching: " << DataSpecUtils::describe(query) << " to route " << DataSpecUtils::describe(route.matcher);
          if (DataSpecUtils::match(route.matcher, query)) {
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
  const char* d = strdup((std::string("name=") + name + "," + channelConfig).c_str());
  spec.options = {
    ConfigParamSpec{"channel-config", VariantType::String, d, {"Out-of-band channel config"}}};
  return spec;
}

} // namespace framework
} // namespace o2
