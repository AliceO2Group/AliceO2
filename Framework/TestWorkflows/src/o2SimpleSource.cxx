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

#include <chrono>
#include <thread>
#include <vector>

#include "Framework/runDataProcessing.h"

using namespace o2::framework;

AlgorithmSpec simplePipe(std::string const& what, int minDelay)
{
  return AlgorithmSpec{[what, minDelay]() {
    srand(getpid());
    return adaptStateless([what, minDelay](DataAllocator& outputs) {
      std::this_thread::sleep_for(std::chrono::seconds((rand() % 5) + minDelay));
      auto& bData = outputs.make<int>(OutputRef{what}, 1);
    });
  }};
}

// This is how you can define your processing in a declarative way
WorkflowSpec defineDataProcessing(ConfigContext const& specs)
{
  return WorkflowSpec{
    {"A",
     Inputs{},
     {OutputSpec{{"a1"}, "TST", "A1"}},
     AlgorithmSpec{adaptStateless(
       [](DataAllocator& outputs) {
         std::this_thread::sleep_for(std::chrono::seconds(rand() % 2));
         auto& aData = outputs.make<int>(OutputRef{"a1"}, 1);
       })},
     {ConfigParamSpec{"some-device-param", VariantType::Int, 1, {"Some device parameter"}}}}};
}
