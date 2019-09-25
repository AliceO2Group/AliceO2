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
#include "Framework/runDataProcessing.h"
#include "Framework/WorkflowSpec.h"

#include <chrono>
#include <vector>

using namespace o2::framework;

AlgorithmSpec simplePipe(std::string const& what, o2::header::DataHeader::SubSpecificationType subSpec, int minDelay)
{
  return AlgorithmSpec{[what, minDelay, subSpec](InitContext& ic) {
    srand(getpid());
    return [what, minDelay, subSpec](ProcessingContext& ctx) {
      std::this_thread::sleep_for(std::chrono::seconds((rand() % 5) + minDelay));
      auto bData = ctx.outputs().make<int>(OutputRef{what, subSpec}, 1);
      bData[0] = subSpec;
    };
  }};
}

// This demonstrates how to use the ``select'' helper to construct inputs for
// you.
WorkflowSpec defineDataProcessing(ConfigContext const& specs)
{
  return WorkflowSpec{
    {"A",
     select(),
     {OutputSpec{{"a1"}, "TST", "A1"},
      OutputSpec{{"a2"}, "TST", "A2"}},
     AlgorithmSpec{
       [](ProcessingContext& ctx) {
         std::this_thread::sleep_for(std::chrono::seconds(rand() % 2));
         auto aData = ctx.outputs().make<int>(OutputRef{"a1"}, 1);
         auto bData = ctx.outputs().make<int>(OutputRef{"a2"}, 1);
       }}},
    {"B",
     select("x:TST/A1/0"), // This will match TST/A1/0 as <origin>/<description>/<subspec> and bind it to x
     {OutputSpec{{"b1"}, "TST", "FORD", 0}},
     simplePipe("b1", 0, 3)},
    {"C",
     select("x:TST/A2/0"),
     Outputs{OutputSpec{{"c1"}, "TST", "FORD", 1}},
     simplePipe("c1", 1, 2)},
    {"D",
     select("x:TST/FORD"),
     Outputs{},
     AlgorithmSpec{
       [](ProcessingContext& ctx) {
         auto& x = ctx.inputs().get<int>("x");
         std::cout << x << std::endl;
       },
     }}};
}
