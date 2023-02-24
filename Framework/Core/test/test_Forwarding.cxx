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
#include "Framework/ControlService.h"
#include "Framework/CallbackService.h"
#include "Framework/EndOfStreamContext.h"
#include "Framework/DeviceSpec.h"
#include "Framework/runDataProcessing.h"
#include "Framework/ControlService.h"

#include <chrono>
#include <vector>

using namespace o2::framework;

AlgorithmSpec simplePipe(std::string const& what, int minDelay)
{
  return AlgorithmSpec{[what, minDelay](InitContext& ic) {
    srand(getpid());
    return [what, minDelay](ProcessingContext& ctx) {
      auto& bData = ctx.outputs().make<int>(OutputRef{what}, 1);
    };
  }};
}

// This is how you can define your processing in a declarative way
WorkflowSpec defineDataProcessing(ConfigContext const& specs)
{
  return WorkflowSpec{
    {"A",
     Inputs{},
     {OutputSpec{{"a1"}, "TST", "A1"}},
     AlgorithmSpec{
       [](ProcessingContext& ctx) {
         auto& aData = ctx.outputs().make<int>(OutputRef{"a1"}, 1);
         ctx.services().get<ControlService>().endOfStream();
         ctx.services().get<ControlService>().readyToQuit(QuitRequest::Me);
       }}},
    {"B",
     {InputSpec{"x", "TST", "A1"}},
     {OutputSpec{{"b1"}, "TST", "B1"}},
     simplePipe("b1", 0)},
    {"C",
     Inputs{InputSpec{"x", "TST", "A1"}},
     Outputs{OutputSpec{{"c1"}, "TST", "C1"}},
     simplePipe("c1", 5)},
    {"D",
     Inputs{
       InputSpec{"b", "TST", "B1"},
       InputSpec{"c", "TST", "C1"}},
     Outputs{},
     AlgorithmSpec{
       adaptStateless([](CallbackService& callbacks) {
         callbacks.set<CallbackService::Id::EndOfStream>([](EndOfStreamContext& context) {
           context.services().get<ControlService>().readyToQuit(QuitRequest::All);
         });
       }),
     }}};
}
