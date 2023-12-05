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

#include "Framework/ControlService.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/DataRefUtils.h"
#include "Framework/Lifetime.h"
#include <SimulationDataFormat/ConstMCTruthContainer.h>
#include <SimulationDataFormat/MCCompLabel.h>
#include "Framework/Task.h"
#include "Framework/Logger.h"

using namespace o2::framework;

namespace o2
{

class MCTruthSourceTask : public o2::framework::Task
{
 public:
  MCTruthSourceTask(bool newmctruth) : mNew{newmctruth} {}

  void init(framework::InitContext& ic) override
  {
    LOG(info) << "Initializing MCTruth source";
    mSize = ic.options().get<int>("size");
  }

  void run(framework::ProcessingContext& pc) override
  {
    if (mFinished) {
      return;
    }
    LOG(info) << "Creating MCTruth container";

    using TruthElement = o2::MCCompLabel;
    using Container = dataformats::MCTruthContainer<TruthElement>;
    Container container;
    // create a very large container and stream it to TTree
    for (int i = 0; i < mSize; ++i) {
      container.addElement(i, TruthElement(i, i, i));
      container.addElement(i, TruthElement(i + 1, i, i));
    }

    if (mNew) {
      LOG(info) << "New serialization";
      // we need to flatten it and write to managed shared memory container
      auto& sharedlabels = pc.outputs().make<o2::dataformats::ConstMCTruthContainer<o2::MCCompLabel>>(Output{"TST", "LABELS", 0});
      container.flatten_to(sharedlabels);
      sleep(1);
    } else {
      LOG(info) << "Old serialization";
      pc.outputs().snapshot({"TST", "LABELS", 0}, container);
      sleep(1);
    }

    // we should be only called once; tell DPL that this process is ready to exit
    pc.services().get<ControlService>().readyToQuit(QuitRequest::Me);
    mFinished = true;
  }

 private:
  bool mFinished = false;
  int mSize = 0;
  bool mNew = false;
  o2::dataformats::MCTruthContainer<long> mLabels; // labels which get filled
};

o2::framework::DataProcessorSpec getMCTruthSourceSpec(bool newmctruth)
{
  // create the full data processor spec using
  //  a name identifier
  //  input description
  //  algorithmic description (here a lambda getting called once to setup the actual processing function)
  //  options that can be used for this processor (here: input file names where to take the hits)
  std::vector<OutputSpec> outputs;
  outputs.emplace_back("TST", "LABELS", 0, Lifetime::Timeframe);

  return DataProcessorSpec{
    "MCTruthSource",
    Inputs{},
    outputs,
    AlgorithmSpec{adaptFromTask<MCTruthSourceTask>(newmctruth)},
    Options{
      {"size", VariantType::Int, 100000, {"Sample size"}}}};
}

} // end namespace o2
