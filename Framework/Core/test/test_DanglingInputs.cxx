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
#include "Framework/DeviceSpec.h"
#include "Framework/runDataProcessing.h"
#include "Framework/ControlService.h"

#include <vector>

using namespace o2::framework;

AlgorithmSpec simplePipe(std::string const& what, int minDelay)
{
  return AlgorithmSpec{[what, minDelay](InitContext& ic) {
    srand(getpid());
    return [what, minDelay](ProcessingContext& ctx) {
      auto bData = ctx.outputs().make<int>(OutputRef{what}, 1);
    };
  }};
}

// This is how you can define your processing in a declarative way
WorkflowSpec defineDataProcessing(ConfigContext const& specs)
{
  return WorkflowSpec{
    {"A",
     Inputs{},
     {OutputSpec{{"a1"}, "TST", "A1"},
      OutputSpec{{"a2"}, "TST", "A2"}},
     AlgorithmSpec{
       [](ProcessingContext& ctx) {
         std::this_thread::sleep_for(std::chrono::milliseconds((rand() % 2 + 1) * 1000));
         auto aData = ctx.outputs().make<int>(OutputRef{"a1"}, 1);
         auto bData = ctx.outputs().make<int>(OutputRef{"a2"}, 1);
       }}},
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
       InputSpec{"timer", "TST", "TIMER", 0, Lifetime::Timer}},
     Outputs{},
     AlgorithmSpec{
       [](ProcessingContext& ctx) {
         ctx.services().get<ControlService>().readyToQuit(QuitRequest::All);
       },
     }}};
}
