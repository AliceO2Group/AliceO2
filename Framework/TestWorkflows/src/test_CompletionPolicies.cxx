// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "Framework/CompletionPolicy.h"
#include "Framework/CompletionPolicyHelpers.h"
#include "Framework/DeviceSpec.h"

#include <cassert>
#include <chrono>
#include <vector>

using namespace o2::framework;

void customize(std::vector<CompletionPolicy>& policies)
{
  std::vector<CompletionPolicy> result{
    CompletionPolicyHelpers::defineByName("discard", CompletionPolicy::CompletionOp::Discard),
    CompletionPolicyHelpers::defineByName("process", CompletionPolicy::CompletionOp::Process),
    CompletionPolicyHelpers::defineByName("wait", CompletionPolicy::CompletionOp::Wait),
    CompletionPolicyHelpers::defineByName("consume", CompletionPolicy::CompletionOp::Consume)};
  policies.swap(result);
}

#include "Framework/runDataProcessing.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/Logger.h"
#include "Framework/ParallelContext.h"
#include "Framework/ControlService.h"
#include <vector>

// This is a simple consumer / producer workflow where both are
// stateful, i.e. they have context which comes from their initialization.
WorkflowSpec defineDataProcessing(ConfigContext const& config)
{
  return WorkflowSpec{
    DataProcessorSpec{
      "hearthbeat",
      {},
      {OutputSpec{{"out1"}, "TST", "TST", 0},
       OutputSpec{{"out2"}, "TST", "TST", 1}},
      AlgorithmSpec{
        [](ProcessingContext& ctx) {
          // We deliberately make only out1 to test that
          // the policies for the following dataprocessors are
          // actually respected.
          ctx.outputs().make<int>(OutputRef{"out1"}, 1);
          std::this_thread::sleep_for(std::chrono::seconds(1));
        }}},
    DataProcessorSpec{
      "discard",
      {
        InputSpec{"in1", "TST", "TST", 0},
        InputSpec{"in2", "TST", "TST", 1},
      },
      {},
      AlgorithmSpec{
        [](ProcessingContext& ctx) {
          LOG(ERROR) << "Should have not been invoked";
          // We deliberately make only out1 to test that
          // the policies for the following dataprocessors are
          // actually respected.
        }}},
    DataProcessorSpec{
      "does-use",
      {InputSpec{"in1", "TST", "TST", 0}},
      {},
      AlgorithmSpec{
        [](ProcessingContext& ctx) {
          // Since this shares a dependency with "discard",
          // it should be forwarded the messages as soon as the former
          // discards them.
        }}},
  };
}
