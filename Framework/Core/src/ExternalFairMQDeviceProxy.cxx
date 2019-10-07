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

namespace o2
{
namespace framework
{

using DataHeader = o2::header::DataHeader;

bool isBroadcastChannel(std::string const& channel)
{
  if (strncmp(channel.data(), "from_", 5) != 0) {
    return false;
  }
  /// FIXME: until we find a better way to avoid using fake input channels
  if (strncmp(channel.data(), "from_internal-dpl-", 18) == 0) {
    return false;
  }
  return true;
}

void sendOnChannel(FairMQDevice& device, FairMQParts&& messages, OutputSpec const& spec, ChannelRetreiver& channelRetreiver)
{
  // Note: DPL is only setting up one instance of a channel while FairMQ allows to have an
  // array of channels, the index is 0 in the call
  auto channel = channelRetreiver(spec);
  constexpr auto index = 0;
  if (channel.empty()) {
    LOG(ERROR) << "can not find matching channel";
    return;
  }
  device.Send(messages, channel, index);
}

void sendOnChannel(FairMQDevice& device, o2::header::Stack&& headerStack, FairMQMessagePtr&& payloadMessage, OutputSpec const& spec, ChannelRetreiver& channelRetreiver)
{
  auto channelName = channelRetreiver(spec);
  constexpr auto index = 0;
  if (channelName.empty()) {
    LOG(ERROR) << "can not find matching channel";
    return;
  }
  for (auto& channelInfo : device.fChannels) {
    if (channelInfo.first != channelName) {
      continue;
    }
    assert(channelInfo.second.size() == 1);
    // FIXME: this assumes there is only one output from here... This should
    //        really do the matchmaking between inputs and output channels.
    auto channelAlloc = o2::pmr::getTransportAllocator(channelInfo.second[index].Transport());
    FairMQMessagePtr headerMessage = o2::pmr::getMessage(std::move(headerStack), channelAlloc);

    FairMQParts out;
    out.AddPart(std::move(headerMessage));
    out.AddPart(std::move(payloadMessage));
    device.Send(out, channelName, 0);
    break;
  }
}

void sendOnChannel(FairMQDevice& device, FairMQMessagePtr&& headerMessage, FairMQMessagePtr&& payloadMessage, OutputSpec const& spec, ChannelRetreiver& channelRetreiver)
{
  FairMQParts out;
  out.AddPart(std::move(headerMessage));
  out.AddPart(std::move(payloadMessage));
  sendOnChannel(device, std::move(out), spec, channelRetreiver);
}

InjectorFunction o2DataModelAdaptor(OutputSpec const& spec, uint64_t startTime, uint64_t step)
{
  auto timesliceId = std::make_shared<size_t>(startTime);
  return [timesliceId, step, spec](FairMQDevice& device, FairMQParts& parts, ChannelRetreiver channelRetreiver) {
    for (size_t i = 0; i < parts.Size() / 2; ++i) {
      auto dh = o2::header::get<DataHeader*>(parts.At(i * 2)->GetData());

      DataProcessingHeader dph{*timesliceId, 0};
      o2::header::Stack headerStack{*dh, dph};
      sendOnChannel(device, std::move(headerStack), std::move(parts.At(i * 2 + 1)), spec, channelRetreiver);
      auto oldTimesliceId = *timesliceId;
      *timesliceId += 1;
    }
  };
}

InjectorFunction dplModelAdaptor(OutputSpec const& spec)
{
  return [spec](FairMQDevice& device, FairMQParts& parts, ChannelRetreiver channelRetreiver) {
    for (size_t i = 0; i < parts.Size() / 2; ++i) {
      sendOnChannel(device, std::move(parts.At(i * 2)), std::move(parts.At(i * 2 + 1)), spec, channelRetreiver);
    }
  };
}

InjectorFunction incrementalConverter(OutputSpec const& spec, uint64_t startTime, uint64_t step)
{
  auto timesliceId = std::make_shared<size_t>(startTime);

  return [timesliceId, spec, step](FairMQDevice& device, FairMQParts& parts, ChannelRetreiver channelRetreiver) {
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

      sendOnChannel(device, std::move(headerStack), std::move(parts.At(i)), spec, channelRetreiver);
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
    auto outputRoutes = ctx.services().get<RawDeviceService>().spec().outputs;
    assert(device);

    // Converter should pump messages
    auto handler = [device, converter, outputRoutes = std::move(outputRoutes)](FairMQParts& inputs, int) {
      auto channelRetreiver = [outputRoutes = std::move(outputRoutes)](OutputSpec const& query) {
        for (auto& route : outputRoutes) {
          if (DataSpecUtils::match(route.matcher, query)) {
            return route.channel;
          }
        }
        return std::string("");
      };
      converter(*device, inputs, channelRetreiver);
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
