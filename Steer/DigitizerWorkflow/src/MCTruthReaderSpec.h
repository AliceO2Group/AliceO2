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

#ifndef STEER_DIGITIZERWORKFLOW_SRC_MCTRUTHREADERSPEC_H_
#define STEER_DIGITIZERWORKFLOW_SRC_MCTRUTHREADERSPEC_H_

#include "Framework/ControlService.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/DataRefUtils.h"
#include "Framework/Lifetime.h"
#include <SimulationDataFormat/ConstMCTruthContainer.h>
#include <SimulationDataFormat/MCCompLabel.h>
#include "SimulationDataFormat/IOMCTruthContainerView.h"
#include "Framework/Task.h"
#include "Framework/Logger.h"
#include "TTree.h"
#include "TFile.h"
#include "TString.h"
#include <sstream>

using namespace o2::framework;

namespace o2
{

class MCTruthReaderTask : public o2::framework::Task
{
 public:
  MCTruthReaderTask(bool newmctruth) : mNew{newmctruth} {}

  void init(framework::InitContext& ic) override
  {
    LOG(info) << "Initializing MCTruth reader ";
  }

  void run(framework::ProcessingContext& pc) override
  {
    if (mFinished) {
      return;
    }
    LOG(info) << "Running MCTruth reader ";
    auto labelfilename = pc.inputs().get<TString*>("trigger");
    LOG(info) << "Opening file " << labelfilename->Data();
    TFile f(labelfilename->Data(), "OPEN");
    auto tree = (TTree*)f.Get("o2sim");
    auto br = tree->GetBranch("Labels");

    if (mNew) {
      //
      dataformats::IOMCTruthContainerView* iocontainer = nullptr;
      br->SetAddress(&iocontainer);
      br->GetEntry(0);

      // publish the labels in a const shared memory container
      auto& sharedlabels = pc.outputs().make<o2::dataformats::ConstMCTruthContainer<o2::MCCompLabel>>(Output{"TST", "LABELS2", 0});
      iocontainer->copyandflatten(sharedlabels);

    } else {
      // the original way with the MCTruthContainer
      dataformats::MCTruthContainer<MCCompLabel>* mccontainer = nullptr;
      br->SetAddress(&mccontainer);
      br->GetEntry(0);

      LOG(info) << "MCCONTAINER CHECK" << mccontainer;
      LOG(info) << "MCCONTAINER CHECK" << mccontainer->getNElements();

      // publish the original labels
      pc.outputs().snapshot(Output{"TST", "LABELS2", 0}, *mccontainer);
    }
    // we should be only called once; tell DPL that this process is ready to exit
    pc.services().get<ControlService>().readyToQuit(QuitRequest::Me);
    mFinished = true;
  }

 private:
  bool mFinished = false;
  bool mNew = false;
};

o2::framework::DataProcessorSpec getMCTruthReaderSpec(bool newmctruth)
{
  std::vector<InputSpec> inputs;
  // input to notify that labels can be read
  inputs.emplace_back("trigger", "TST", "TRIGGERREAD", 0, Lifetime::Timeframe);
  return DataProcessorSpec{
    "MCTruthReader",
    inputs,
    Outputs{{"TST", "LABELS2", 0, Lifetime::Timeframe}},
    AlgorithmSpec{adaptFromTask<MCTruthReaderTask>(newmctruth)},
    Options{}};
}

} // end namespace o2

#endif
