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
#include "Framework/ConfigParamSpec.h"
#include "Framework/CompletionPolicyHelpers.h"
#include "Framework/DeviceSpec.h"
#include "Framework/RawDeviceService.h"
#include "Framework/SourceInfoHeader.h"
#include <fairmq/Device.h>

#include <chrono>
#include <thread>
#include <vector>

#include "Framework/runDataProcessing.h"

using namespace o2::framework;

void sendEndOfStream(fair::mq::Device& device, std::string channel)
{
  fair::mq::Parts parts;
  fair::mq::MessagePtr payload(device.NewMessage());
  SourceInfoHeader sih;
  sih.state = InputChannelState::Completed;
  o2::header::DataHeader dh;
  dh.dataOrigin = "TST";
  dh.dataDescription = "A";
  dh.subSpecification = 0;
  dh.payloadSize = 1000;

  DataProcessingHeader dph{1, 0};
  auto channelAlloc = o2::pmr::getTransportAllocator(device.GetChannel(channel, 0).Transport());
  auto header = o2::pmr::getMessage(o2::header::Stack{channelAlloc, dh, dph, sih});
  // sigh... See if we can avoid having it const by not
  // exposing it to the user in the first place.
  parts.AddPart(std::move(header));
  parts.AddPart(std::move(payload));
  device.Send(parts, channel, 0);
}

// This is how you can define your processing in a declarative way
WorkflowSpec defineDataProcessing(ConfigContext const&)
{
  return WorkflowSpec{
    {.name = "A",
     .outputs = {OutputSpec{"TST", "A", 0, Lifetime::OutOfBand}},
     .algorithm = AlgorithmSpec{adaptStateless(
       [](RawDeviceService& service) {
         for (auto& channel : service.device()->GetChannels()) {
           LOG(info) << channel.first;
         }
         std::this_thread::sleep_for(std::chrono::seconds(rand() % 2));
         auto msg = service.device()->NewMessageFor("downstream", 0, 1000);

         o2::header::DataHeader dh;
         dh.dataOrigin = "TST";
         dh.dataDescription = "A";
         dh.subSpecification = 0;
         dh.payloadSize = 1000;

         DataProcessingHeader dph{1, 0};
         // we have to move the incoming data
         o2::header::Stack headerStack{dh, dph};

         auto channelAlloc = o2::pmr::getTransportAllocator(service.device()->GetChannels()["downstream"][0].Transport());
         fair::mq::MessagePtr headerMessage = o2::pmr::getMessage(std::move(headerStack), channelAlloc);

         fair::mq::Parts out;
         out.AddPart(std::move(headerMessage));
         out.AddPart(std::move(msg));
         o2::header::hexDump("header", out.At(0)->GetData(), out.At(0)->GetSize(), 100);

         service.device()->Send(out, "downstream", 0);
         sendEndOfStream(*service.device(), "downstream");
         exit(1);
       })}}};
}
