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
#include "Framework/CompletionPolicy.h"
#include "Framework/DeviceSpec.h"
#include <InfoLogger/InfoLogger.hxx>

#include <chrono>
#include <vector>

using namespace o2::framework;
using namespace AliceO2::InfoLogger;

void customize(std::vector<ConfigParamSpec>& options)
{
  options.push_back(ConfigParamSpec{"anInt", VariantType::Int, 1, {"an int option"}});
  options.push_back(ConfigParamSpec{"aFloat", VariantType::Float, 2.0f, {"a float option"}});
  options.push_back(ConfigParamSpec{"aDouble", VariantType::Double, 3., {"a double option"}});
  options.push_back(ConfigParamSpec{"aString", VariantType::String, "foo", {"a string option"}});
  options.push_back(ConfigParamSpec{"aBool", VariantType::Bool, true, {"a boolean option"}});
}

// This completion policy will only be applied to the device called `D` and
// will process an InputRecord which had any of its constituent updated.
void customize(std::vector<CompletionPolicy>& policies)
{
  auto matcher = [](DeviceSpec const& device) -> bool {
    return device.name == "D";
  };
  auto policy = [](gsl::span<PartRef const> const& inputs) -> CompletionPolicy::CompletionOp {
    return CompletionPolicy::CompletionOp::Process;
  };
  policies.push_back({CompletionPolicy{"process-any", matcher, policy}});
}

#include "Framework/runDataProcessing.h"

AlgorithmSpec simplePipe(std::string const& what, int minDelay)
{
  return AlgorithmSpec{adaptStateful([what, minDelay]() {
    srand(getpid());
    return adaptStateless([what, minDelay](DataAllocator& outputs) {
      std::this_thread::sleep_for(std::chrono::seconds((rand() % 5) + minDelay));
      auto bData = outputs.make<int>(OutputRef{what}, 1);
    });
  })};
}

// This is how you can define your processing in a declarative way
WorkflowSpec defineDataProcessing(ConfigContext const& specs)
{
  return WorkflowSpec{
    {"A",
     Inputs{},
     {OutputSpec{{"a1"}, "TST", "A1"},
      OutputSpec{{"a2"}, "TST", "A2"}},
     AlgorithmSpec{adaptStateless(
       [](DataAllocator& outputs, InfoLogger& logger) {
         std::this_thread::sleep_for(std::chrono::seconds(rand() % 2));
         auto aData = outputs.make<int>(OutputRef{"a1"}, 1);
         auto bData = outputs.make<int>(OutputRef{"a2"}, 1);
         logger.log("This goes to infologger");
       })}},
    {"B",
     {InputSpec{"x", "TST", "A1"}},
     {OutputSpec{{"b1"}, "TST", "B1"}},
     simplePipe("b1", 0)},
    {"C",
     Inputs{InputSpec{"x", "TST", "A2"}},
     Outputs{OutputSpec{{"c1"}, "TST", "C1"}},
     simplePipe("c1", 5)},
    {"D",
     Inputs{
       InputSpec{"b", "TST", "B1"},
       InputSpec{"c", "TST", "C1"},
     },
     Outputs{},
     AlgorithmSpec{adaptStateless([]() {})}}};
}
