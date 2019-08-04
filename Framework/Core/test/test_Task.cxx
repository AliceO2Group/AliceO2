// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#include "Framework/runDataProcessing.h"
#include "Framework/Task.h"
#include "Framework/ControlService.h"
#include <Monitoring/Monitoring.h>
#include "Framework/ControlService.h"

#define ASSERT_ERROR(condition)                                                                      \
  if ((condition) == false) {                                                                        \
    LOG(ERROR) << R"(Test condition ")" #condition R"(" failed at )" << __FILE__ << ":" << __LINE__; \
  }

using namespace o2::framework;

// This is a stateful task, where we send the state downstream.
class ATask : public Task
{
 public:
  ATask(int state)
    : mSomeState{state} {}
  void init(InitContext& ic) final
  {
    mSomeState += 1;
  }
  void run(ProcessingContext& pc) final
  {
    auto result = pc.outputs().make<int>({"dummy"}, 1);
    result[0] = mSomeState;
    pc.services().get<o2::monitoring::Monitoring>().send({result[0], "output"});
    pc.services().get<ControlService>().readyToQuit(false);
  }

 private:
  int mSomeState;
};

// This is a stateless sink, where we verify that the state
// we receive from ATask is the one we expected.
class BTask : public Task
{
 public:
  void run(ProcessingContext& pc) final
  {
    auto result = pc.inputs().get<int>("in");
    ASSERT_ERROR(result == 2);
    pc.services().get<o2::monitoring::Monitoring>().send({result, "input"});
    pc.services().get<ControlService>().readyToQuit(true);
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const&)
{
  return WorkflowSpec{
    DataProcessorSpec{
      "producer",
      Inputs{},
      {
        OutputSpec{{"dummy"}, "TST", "TEST"},
      },
      adaptFromTask<ATask>(1)},
    DataProcessorSpec{
      "consumer",
      Inputs{
        InputSpec{"in", "TST", "TEST"},
      },
      {},
      adaptFromTask<BTask>()}};
}
