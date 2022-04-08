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
#include "Framework/DataTakingContext.h"
#include "Framework/DeviceSpec.h"
#include "Framework/ConfigContext.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/ControlService.h"
#include "Framework/Task.h"
#include "Framework/AlgorithmSpec.h"

#include <iostream>
#include <vector>

using namespace o2::framework;

// we need to add workflow options before including Framework/runDataProcessing
void customize(std::vector<o2::framework::ConfigParamSpec>& workflowOptions)
{
  // option to disable MC truth
  workflowOptions.push_back(ConfigParamSpec{"combined-source", o2::framework::VariantType::Bool, false, {"combines source devices into 1 DPL process"}});
}

#include "Framework/runDataProcessing.h"

// a utility combining multiple specs into one
// (with some checking that it makes sense)
// spits out the combined spec (merged input/outchannels and AlgorithmSpec)
// Can put policies later whether to multi-thread or serialized internally etc.
auto specCombiner = [](std::string const& name, std::vector<DataProcessorSpec> const& speccollection) {
  std::vector<OutputSpec> combinedOutputSpec;
  std::vector<InputSpec> combinedInputSpec;
  // std::vector<> combinedOptions; --> to be done
  for (auto& spec : speccollection) {
    // merge input specs
    for (auto& is : spec.inputs) {
      combinedInputSpec.push_back(is);
    }
    // merge output specs
    for (auto& os : spec.outputs) {
      combinedOutputSpec.push_back(os);
    }
  }

  // logic for combined task processing function --> target is to run one only
  class CombinedTask
  {
   public:
    CombinedTask(std::vector<DataProcessorSpec> const& s) : tasks(s){};

    void init(o2::framework::InitContext& ic)
    {
      std::cerr << "Init Combined\n";
      for (auto& t : tasks) {
        // the init function actually creates the onProcess function
        // which we have to do here (maybe some more stuff needed)
        t.algorithm.onProcess = t.algorithm.onInit(ic);
      }
    }

    void run(o2::framework::ProcessingContext& pc)
    {
      std::cerr << "Processing Combined\n";
      for (auto& t : tasks) {
        t.algorithm.onProcess(pc);
      }
    }

   private:
    std::vector<DataProcessorSpec> tasks;
  };

  return DataProcessorSpec{
    name,
    combinedInputSpec,
    combinedOutputSpec,
    AlgorithmSpec{adaptFromTask<CombinedTask>(speccollection)},
    {}
    /* a couple of other fields can be set ... */
  };
};

WorkflowSpec defineDataProcessing(ConfigContext const& configc)
{
  // very simple source task
  class TaskA
  {
   public:
    void init(o2::framework::InitContext&) { std::cout << "Init A\n"; }
    void run(o2::framework::ProcessingContext& pc)
    {
      std::cout << "Processing A\n";
      int a = 110;
      pc.outputs().snapshot({"SIM", "TaskA", 0, Lifetime::Timeframe}, a);
      pc.services().get<ControlService>().endOfStream();
    }
  };

  // very simple source task
  class TaskB
  {
   public:
    void init(o2::framework::InitContext&) { std::cout << "Init B\n"; }
    void run(o2::framework::ProcessingContext& pc)
    {
      std::cout << "Processing B\n";
      int b = 222;
      pc.outputs().snapshot({"SIM", "TaskB", 0, Lifetime::Timeframe}, b);
      pc.services().get<ControlService>().endOfStream();
    }
  };

  // very simple consumer task
  class TaskC
  {
   public:
    void init(o2::framework::InitContext&) { std::cout << "Init C\n"; }
    void run(o2::framework::ProcessingContext& pc)
    {
      // we take the input from A + B and sum together
      auto a = pc.inputs().get<int>("FromTaskA");
      auto b = pc.inputs().get<int>("FromTaskB");
      std::cout << "Processing C : result " << a + b << "\n";
      pc.services().get<ControlService>().endOfStream();
    }
  };

  DataProcessorSpec SpecA{
    "TaskA",
    {}, /* input */
    {OutputSpec{{"FromTaskA"}, "SIM", "TaskA", 0, Lifetime::Timeframe}},
    AlgorithmSpec{adaptFromTask<TaskA>()},
    {}};

  DataProcessorSpec SpecB{
    "TaskB",
    {}, /* input */
    {OutputSpec{{"FromTaskB"}, "SIM", "TaskB", 0, Lifetime::Timeframe}},
    AlgorithmSpec{adaptFromTask<TaskB>()},
    {}};

  DataProcessorSpec SpecC{
    "TaskC",
    {
      {"FromTaskA", "SIM", "TaskA", 0, Lifetime::Timeframe},
      {"FromTaskB", "SIM", "TaskB", 0, Lifetime::Timeframe},
    },  /* input */
    {}, /* output */
    AlgorithmSpec{adaptFromTask<TaskC>()},
    {}};

  WorkflowSpec specs;
  if (configc.options().get<bool>("combined-source")) {
    // merge source devices into one
    specs.push_back(specCombiner("CombinedTaskATaskB", {SpecA, SpecB}));
  } else {
    // treat sources individually
    specs.push_back(SpecA);
    specs.push_back(SpecB);
  }
  // put consumer
  specs.push_back(SpecC);
  return specs;
}
