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
#include "Framework/ProcessingContext.h"
#include "Framework/RawDeviceService.h"
#include "Framework/InitContext.h"
#include "Framework/ConfigParamSpec.h"
#include "Framework/AlgorithmSpec.h"
#include "Headers/DataHeader.h"
#include "Framework/DataProcessingHeader.h"
#include <fairmq/FairMQParts.h>
#include <cstring>
#include <cassert>
#include <memory>

namespace o2 {
namespace framework {

using DataHeader = o2::header::DataHeader;

void broadcastMessage(FairMQDevice &device, o2::header::Stack &&headerStack, FairMQMessagePtr &&payloadMessage, int index) {
  for (auto &channelInfo : device.fChannels) {
    auto channel = channelInfo.first;
    assert(channelInfo.second.size() == 1);
    // FIXME: I need to make sure the input channel is not used... For the moment 
    //        I rely on the convention dpl channels start with "from_".
    if (strncmp(channel.data(), "from_", 5) != 0) {
      continue;
    }

    // FIXME: this assumes there is only one output from here... This should
    //        really do the matchmaking between inputs and output channels.
    FairMQMessagePtr headerMessage = device.NewMessageFor(channel, index,
                                                          headerStack.buffer.get(),
                                                          headerStack.bufferSize,
                                                          &o2::header::Stack::freefn,
                                                          headerStack.buffer.get());
    headerStack.buffer.release();
    FairMQParts out;
    out.AddPart(std::move(headerMessage));
    out.AddPart(std::move(payloadMessage));
    device.Send(out, channel, index);
  }
}

// FIXME: should I filter out only the output specs which match?
InjectorFunction o2DataModelAdaptor(OutputSpec const &spec, uint64_t startTime, uint64_t step) {
  auto timesliceId = std::make_shared<size_t>(startTime);
  return [timesliceId, step](FairMQDevice &device, FairMQParts &parts, int index) {
    for (size_t i = 0; i < parts.Size()/2; ++i) {
      auto dh = o2::header::get<DataHeader>(parts.At(i*2)->GetData());

      DataProcessingHeader dph{*timesliceId, 0};
      o2::header::Stack headerStack{*dh, dph};
      broadcastMessage(device, std::move(headerStack), std::move(parts.At(i*2+1)), index);
      auto oldTimesliceId = *timesliceId;
      *timesliceId += 1;
    }
  };
}

InjectorFunction incrementalConverter(OutputSpec const &spec, uint64_t startTime, uint64_t step) {
  auto timesliceId = std::make_shared<size_t>(startTime);

  return [timesliceId, spec, step](FairMQDevice &device, FairMQParts &parts, int index) {
    // We iterate on all the parts and we send them two by two,
    // adding the appropriate O2 header.
    for (size_t i = 0; i < parts.Size(); ++i) {
      DataHeader dh;
      dh.dataOrigin = spec.origin;
      dh.dataDescription = spec.description;
      dh.subSpecification = spec.subSpec;
      dh.payloadSize = parts.At(i)->GetSize();

      DataProcessingHeader dph{*timesliceId, 0};
      *timesliceId += step;
      //we have to move the incoming data
      o2::header::Stack headerStack{dh, dph};

      broadcastMessage(device, std::move(headerStack), std::move(parts.At(i)), index);
    }
  };
}

DataProcessorSpec specifyExternalFairMQDeviceProxy(char const *name,
                                  std::vector<OutputSpec> const &outputs,
                                  char const *channelConfig,
                                  std::function<void(FairMQDevice &device,
                                                     FairMQParts& inputs,
                                                     int index)> converter) {
  DataProcessorSpec spec;
  spec.name = strdup(name);
  spec.inputs = {};
  spec.outputs = outputs;
  // The Init method will register a new "Out of band" channel and 
  // attach an OnData to it which is responsible for converting incoming
  // messages into DPL messages.
  spec.algorithm = AlgorithmSpec{[converter, channel = spec.name](InitContext &ctx) {
      auto device = ctx.services().get<RawDeviceService>().device();
      assert(device);

      // Converter should pump messages
      auto handler = [device,converter](FairMQParts &inputs, int idx) {
        converter(*device, inputs, idx);
        return true;
      };

      device->OnData(channel, handler);
      return [](ProcessingContext &) {};
    }
  };
  const char *d = strdup((std::string("name=") + name + "," + channelConfig).c_str());
  spec.options = {
    ConfigParamSpec{"channel-config", VariantType::String, d, {"Out-of-band channel config"}}
  };
  return spec;
}

} // namespace framework
} // namespace o2
