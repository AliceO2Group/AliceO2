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
#include "Framework/ChannelParamSpec.h"

#include <chrono>
#include <thread>
#include <vector>
#include <fairmq/Device.h>
#include "Framework/runDataProcessing.h"

using namespace o2::framework;

AlgorithmSpec simplePipe(std::string const& what, int minDelay)
{
  return AlgorithmSpec{adaptStateful([what, minDelay]() {
    srand(getpid());
    return adaptStateless([what, minDelay](RawDeviceService& device) {
      std::unique_ptr<fair::mq::Message> msg;
      device.device()->Receive(msg, "upstream", 0);
      LOGP(info, "Callback invoked. Size of the message {}", msg->GetSize());

      device.device()->WaitFor(std::chrono::seconds(minDelay));
    });
  })};
}

// This is how you can define your processing in a declarative way
WorkflowSpec defineDataProcessing(ConfigContext const&)
{
  return WorkflowSpec{
    {.name = "B",
     .inputs = {{{"external"}, "TST", "EXT", 0, Lifetime::OutOfBand, channelParamSpec("upstream")}},
     .algorithm = simplePipe("b1", 0)},
  };
}
