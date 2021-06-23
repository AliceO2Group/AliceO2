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
#include "Framework/DeviceSpec.h"
#include "Framework/runDataProcessing.h"
#include "Framework/ControlService.h"

#include <chrono>
#include <thread>
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

// a1 is not actually used by anything, however it might.
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
         auto& aData1 = ctx.outputs().make<int>(OutputRef{"a1"}, 1);
         auto& aData2 = ctx.outputs().make<int>(Output{"TST", "A2"}, 1);
         ctx.services().get<ControlService>().endOfStream();
       }}},
    {"B",
     {InputSpec{{"a1"}, "TST", "A1"}},
     {},
     AlgorithmSpec{
       [](ProcessingContext& ctx) {
       }}}};
}
