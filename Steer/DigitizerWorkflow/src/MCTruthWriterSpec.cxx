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

class MCTruthWriterTask : public o2::framework::Task
{
 public:
  MCTruthWriterTask(int id, bool doio, bool newmctruth) : mID{id}, mIO{doio}, mNew{newmctruth} {}

  void init(framework::InitContext& ic) override
  {
    LOG(info) << "Initializing MCTruth consumer " << mID;
  }

  void run(framework::ProcessingContext& pc) override
  {
    if (mFinished) {
      return;
    }
    LOG(info) << "Running MCTruth consumer " << mID;
    TString labelfilename;
    if (mNew) {
      auto labels = pc.inputs().get<o2::dataformats::ConstMCTruthContainer<o2::MCCompLabel>>(mID >= 0 ? "labels" : "labels2");
      LOG(info) << "GOT " << labels.getNElements() << " labels";

      sleep(1);

      if (mIO) {
        dataformats::IOMCTruthContainerView io(labels);
        labelfilename = "labels_new.root";
        TFile f(labelfilename.Data(), "RECREATE");
        TTree tree("o2sim", "o2sim");
        auto br = tree.Branch("Labels", &io);
        tree.Fill();
        f.Write();
        f.Close();
      }

      sleep(1);
    } else {
      auto labels = pc.inputs().get<o2::dataformats::MCTruthContainer<o2::MCCompLabel>*>(mID >= 0 ? "labels" : "labels2");
      LOG(info) << "GOT " << labels->getNElements() << " labels";

      sleep(1);

      if (mIO) {
        labelfilename = "labels_old.root";
        TFile f(labelfilename.Data(), "RECREATE");
        TTree tree("o2sim", "o2sim");
        auto rawptr = labels.get();
        auto br = tree.Branch("Labels", &rawptr);
        tree.Fill();
        f.Write();
        f.Close();
      }
      sleep(1);
    }
    if (mIO) {
      // this triggers the reader process
      pc.outputs().snapshot({"TST", "TRIGGERREAD", 0}, labelfilename);
    }

    // we should be only called once; tell DPL that this process is ready to exit
    pc.services().get<ControlService>().readyToQuit(QuitRequest::Me);
    mFinished = true;
  }

 private:
  bool mFinished = false;
  bool mNew = false;
  bool mIO = false;
  int mID = 0;
  o2::dataformats::MCTruthContainer<long> mLabels; // labels which get filled
};

o2::framework::DataProcessorSpec getMCTruthWriterSpec(int id, bool doio, bool newmctruth)
{
  std::vector<InputSpec> inputs;
  if (id == -1) {
    // we use this id as a secondary consumer of the LABELS2 channel
    inputs.emplace_back("labels2", "TST", "LABELS2", 0, Lifetime::Timeframe);
  } else {
    inputs.emplace_back("labels", "TST", "LABELS", 0, Lifetime::Timeframe);
  }
  std::stringstream str;
  str << "MCTruthWriter" << id;

  std::vector<OutputSpec> outputs;
  if (doio) {
    outputs.emplace_back("TST", "TRIGGERREAD", 0, Lifetime::Timeframe);
  }
  return DataProcessorSpec{
    str.str(),
    inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<MCTruthWriterTask>(id, doio, newmctruth)},
    Options{}};
}

} // end namespace o2
