// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#include "Framework/ConfigParamSpec.h"
#include "Framework/CompletionPolicyHelpers.h"
#include "Framework/DeviceSpec.h"
#include "Framework/RawDeviceService.h"

#include <chrono>
#include <thread>
#include <vector>
#include <FairMQDevice.h>
#include "Framework/runDataProcessing.h"

using namespace o2::framework;

AlgorithmSpec simplePipe(std::string const& what, int minDelay)
{
  return AlgorithmSpec{adaptStateful([what, minDelay]() {
    srand(getpid());
    return adaptStateless([what, minDelay](DataAllocator& outputs, RawDeviceService& device) {
      device.device()->WaitFor(std::chrono::seconds(minDelay));
      auto& bData = outputs.make<int>(OutputRef{what}, 1);
    });
  })};
}

// This is how you can define your processing in a declarative way
WorkflowSpec defineDataProcessing(ConfigContext const& specs)
{
  return WorkflowSpec{
    {"B",
     {},
     {OutputSpec{{"b1"}, "TST", "B1"}},
     simplePipe("b1", 0)},
    {"D",
     Inputs{
       InputSpec{"b", "TST", "B1"},
     },
     Outputs{},
     AlgorithmSpec{adaptStateless([]() {})}}};
}
