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
#include <vector>
using namespace o2::framework;
void customize(std::vector<ConfigParamSpec> &options) {
  options.push_back(o2::framework::ConfigParamSpec{"anInt", VariantType::Int, 1, {"an int option"}});
  options.push_back(o2::framework::ConfigParamSpec{"aFloat", VariantType::Float, 2.0f, {"a float option"}});
  options.push_back(o2::framework::ConfigParamSpec{"aDouble", VariantType::Double, 3., {"a double option"}});
  options.push_back(o2::framework::ConfigParamSpec{"aString", VariantType::String, "foo", {"a string option"}});
  options.push_back(o2::framework::ConfigParamSpec{"aBool", VariantType::Bool, true, {"a boolean option"}});
};
#include "Framework/runDataProcessing.h"


AlgorithmSpec simplePipe(std::string const &what) {
  return AlgorithmSpec{ [what](InitContext& ic) {
    srand(getpid());
    return [what](ProcessingContext& ctx) {
      sleep(rand() % 5);
      auto bData = ctx.outputs().make<int>(OutputRef{ what }, 1);
    };
  } };
}

// This is how you can define your processing in a declarative way
WorkflowSpec defineDataProcessing(ConfigContext const&specs) {
  return WorkflowSpec{
    { "A",
      Inputs{},
      { OutputSpec{ { "a1" }, "TST", "A1" },
        OutputSpec{ { "a2" }, "TST", "A2" } },
      AlgorithmSpec{
        [](ProcessingContext& ctx) {
          sleep(rand() % 5);
          auto aData = ctx.outputs().make<int>(OutputRef{ "a1" }, 1);
          auto bData = ctx.outputs().make<int>(OutputRef{ "a2" }, 1);
        } } },
    { "B",
      { InputSpec{ "x", "TST", "A1" } },
      { OutputSpec{ { "b1" }, "TST", "B1" } },
      simplePipe("b1") },
    { "C",
      Inputs{ InputSpec{ "x", "TST", "A2" } },
      Outputs{ OutputSpec{ { "c1" }, "TST", "C1" } },
      simplePipe("c1") },
    { "D",
      Inputs{
        InputSpec{ "b", "TST", "B1" },
        InputSpec{ "c", "TST", "C1" },
      },
      Outputs{},
      AlgorithmSpec{
        [](ProcessingContext& ctx) {
        },
      } }
  };
}
