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

/// @file   NoInpDummyOutSpec.cxx

#include <vector>
#include "GlobalTrackingWorkflowHelpers/NoInpDummyOutSpec.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/Task.h"
#include "Framework/CCDBParamSpec.h"
#include "Framework/ControlService.h"
#include "Framework/DeviceSpec.h"

using namespace o2::framework;

namespace o2
{
namespace globaltracking
{

class NoInpDummyOut : public Task
{
 public:
  NoInpDummyOut() = default;
  ~NoInpDummyOut() override = default;
  void init(InitContext& ic) final;
  void run(ProcessingContext& pc) final;

 private:
  int mLoops = -1;
};

void NoInpDummyOut::init(InitContext& ic)
{
  mLoops = ic.options().get<int>("max-loops");
}

void NoInpDummyOut::run(ProcessingContext& pc)
{
  static int counter = 0;
  // send just once dummy output to trigger the ccdb-fetcher
  pc.outputs().make<std::vector<char>>(Output{"GLO", "DUMMY_OUT", 0});
  if (mLoops >= 0 && ++counter >= mLoops) {
    pc.services().get<o2::framework::ControlService>().endOfStream();
    pc.services().get<o2::framework::ControlService>().readyToQuit(framework::QuitRequest::Me);
  }
}

DataProcessorSpec getNoInpDummyOutSpec(int nloop)
{
  std::vector<OutputSpec> outputs = {{"GLO", "DUMMY_OUT", 0, Lifetime::Timeframe}};
  return DataProcessorSpec{
    "no-inp-dummy-out",
    {},
    outputs,
    AlgorithmSpec{adaptFromTask<NoInpDummyOut>()},
    Options{ConfigParamSpec{"max-loops", VariantType::Int, nloop, {"max number of loops"}}}};
}

} // namespace globaltracking
} // namespace o2
