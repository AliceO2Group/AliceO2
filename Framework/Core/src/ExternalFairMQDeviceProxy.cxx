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

#include "./DeviceSpecHelpers.h"

#include <fairmq/FairMQParts.h>
#include <fairmq/FairMQDevice.h>
#include <cstring>
#include <cassert>
#include <memory>
#include <optional>
#include <unordered_map>
#include <numeric> // std::accumulate

namespace o2
{
namespace framework
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
  }
  if (timeout > 0 && timeout <= maxTimeout) {
    LOG(WARNING) << "dispatching on channel " << channel << "was delayed by " << timeout << " ms";
  }
  // TODO: feeling this is a bit awkward, but the interface of FairMQParts does not provide a
  // method to clear the content.
  // Maybe the FairMQ API can be improved at some point. Actually the ownership of all messages should be passed
  // on to the transport and the messages should be empty after sending and the parts content can be cleared.
  //assert(std::accumulate(messages.begin(), messages.end(), true, [](bool a, auto const& msg) {return a && (msg.get() == nullptr);}));
  messages.fParts.clear();
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
    sendOnChannel(device, out, channelName);
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
    for (size_t msgidx = 0; msgidx < parts.Size() / 2; ++msgidx) {
      if (indicesDone[msgidx]) {
        continue;
      }

      auto dh = o2::header::get<DataHeader*>(parts.At(msgidx * 2)->GetData());
      if (!dh) {
        LOG(ERROR) << "data on input " << msgidx << " does not follow the O2 data model, DataHeader missing";
        continue;
      }
      LOG(DEBUG) << msgidx << ": " << DataSpecUtils::describe(OutputSpec{dh->dataOrigin, dh->dataDescription, dh->subSpecification}) << " part " << dh->splitPayloadIndex << " of " << dh->splitPayloadParts << "  payload " << parts.At(msgidx * 2 + 1)->GetSize();

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
            auto [indexList, payloadSize] = findSplitParts(parts, msgidx, indicesDone);
            for (auto const& partidx : indexList) {
              if (partidx == msgidx) {
                continue;
              }
              auto dh = o2::header::get<DataHeader*>(parts.At(partidx * 2)->GetData());
              if (dh) {
                LOG(DEBUG) << partidx << ": " << DataSpecUtils::describe(OutputSpec{dh->dataOrigin, dh->dataDescription, dh->subSpecification}) << " part " << dh->splitPayloadIndex << " of " << dh->splitPayloadParts << "  payload " << parts.At(partidx * 2 + 1)->GetSize();
              }
            }
            if (payloadSize > 0) {
              size_t headerSize = parts.At(msgidx * 2)->GetSize();
              auto headerMessage = device.NewMessageFor(channelName, 0, headerSize);
              memcpy(headerMessage->GetData(), parts.At(msgidx * 2)->GetData(), headerSize);
              auto clonedh = const_cast<DataHeader*>(o2::header::get<DataHeader*>(headerMessage->GetData()));
              clonedh->splitPayloadParts = 0;
              clonedh->splitPayloadIndex = 0;
              clonedh->payloadSize = payloadSize;
              outputs[channelName].AddPart(std::move(headerMessage));
              outputs[channelName].AddPart(std::move(mergePayloads(device, parts, indexList, payloadSize, channelName)));
              LOG(DEBUG) << "merged " << indexList.size() << " split payload parts in new message of size " << payloadSize << " on channel " << channelName << " (" << outputs[channelName].Size() << " parts)";
            }
          } else {
            outputs[channelName].AddPart(std::move(parts.At(msgidx * 2)));
            outputs[channelName].AddPart(std::move(parts.At(msgidx * 2 + 1)));
            LOG(DEBUG) << "associating part with index " << msgidx << " to channel " << channelName << " (" << outputs[channelName].Size() << ")";
          }
          // TODO: implement configurable dispatch policy preferably via the workflow spec config
          sendOnChannel(device, outputs[channelName], channelName);
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
