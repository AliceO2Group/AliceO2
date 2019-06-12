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

#include <chrono>
#include <vector>

using namespace o2::framework;

AlgorithmSpec simplePipe(std::string const& what, int minDelay)
{
  return AlgorithmSpec{ [what, minDelay](InitContext& ic) {
    srand(getpid());
    return [what, minDelay](ProcessingContext& ctx) {
      auto bData = ctx.outputs().make<int>(OutputRef{ what }, 1);
    };
  } };
}

// a1 is not actually used by anything, however it might.
WorkflowSpec defineDataProcessing(ConfigContext const& specs)
{
  return WorkflowSpec{
    { "A",
      Inputs{},
      { OutputSpec{ { "a1" }, "TST", "A1" },
        OutputSpec{ { "a2" }, "TST", "A2" } },
      AlgorithmSpec{
        [](ProcessingContext& ctx) {
          std::this_thread::sleep_for(std::chrono::milliseconds((rand() % 2 + 1) * 1000));
          auto aData1 = ctx.outputs().make<int>(OutputRef{ "a1" }, 1);
          auto aData2 = ctx.outputs().make<int>(Output{ "TST", "A2" }, 1);
        } } },
    { "B",
      { InputSpec{ { "a1" }, "TST", "A1" } },
      {},
      AlgorithmSpec{
        [](ProcessingContext& ctx) {
          ctx.services().get<ControlService>().readyToQuit(true);
        } } }
  };
}
