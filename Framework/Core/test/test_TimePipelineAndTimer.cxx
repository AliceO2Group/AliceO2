// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "Framework/CompletionPolicyHelpers.h"
using namespace o2::framework;
void customize(std::vector<CompletionPolicy>& policies)
{
  policies.push_back(CompletionPolicyHelpers::defineByName("dataReceiver", CompletionPolicy::CompletionOp::Consume));
}

#include "Framework/runDataProcessing.h"
#include "Framework/ControlService.h"

using namespace o2::framework;

WorkflowSpec defineDataProcessing(ConfigContext const&)
{
  WorkflowSpec specs;

  DataProcessorSpec producer{
    "dataProducer",
    Inputs{},
    Outputs{
      {{"a"}, "A", "A"}},
    AlgorithmSpec{
      [](ProcessingContext& ctx) {
        if (static bool sent = false; !sent) {
          ctx.outputs().make<char>(OutputRef{"a", 0}, 1);
          sent = true;
        }
      }}};

  DataProcessorSpec receiver{
    "dataReceiver",
    Inputs{
      {{"a"}, "A", "A"},
      {{"time"}, "T", "T", 0, Lifetime::Timer}},
    Outputs{},
    AlgorithmSpec{
      [](ProcessingContext& ctx) {
        static int received = 0;

        if (ctx.inputs().isValid("a")) {
          ctx.inputs().get<char>("a");
          received++;
        }

        if (received == 2 && ctx.inputs().isValid("time")) {
          ctx.services().get<ControlService>().readyToQuit(QuitRequest::All);
        }
      }},
    Options{
      {"period-time", VariantType::Int, 2 * 1000000, {"timer period"}}}};

  specs.push_back(timePipeline(producer, 2));
  specs.push_back(receiver);

  return specs;
}
