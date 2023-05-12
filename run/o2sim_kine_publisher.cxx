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

#include "Framework/AnalysisTask.h"
#include "Framework/runDataProcessing.h"
#include "SimulationDataFormat/MCTrack.h"
#include "Steer/MCKinematicsReader.h"

using namespace o2::framework;
using namespace o2::steer;

struct KinePublisher {

  Configurable<std::string> kineFileName{"kineFileName", "o2sim", "name of the _Kine.root file (without '_Kine.root')"};

  int nEvents = 0;
  int event = 0;
  MCKinematicsReader* mcKinReader = new o2::steer::MCKinematicsReader();

  void init(o2::framework::InitContext& ic)
  {
    mcKinReader->initFromKinematics(std::string(kineFileName));
    nEvents = mcKinReader->getNEvents(0);
  }

  void run(o2::framework::ProcessingContext& pc)
  {
    auto mcevent = mcKinReader->getMCEventHeader(0, event);
    auto mctracks = mcKinReader->getTracks(0, event);
    pc.outputs().snapshot(Output{"MC", "MCHEADER", 0, Lifetime::Timeframe}, mcevent);
    pc.outputs().snapshot(Output{"MC", "MCTRACKS", 0, Lifetime::Timeframe}, mctracks);
    event++;
    if (event >= nEvents) {
      pc.services().get<ControlService>().endOfStream();
      pc.services().get<ControlService>().readyToQuit(QuitRequest::Me);
    }
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  WorkflowSpec specs;
  std::vector<OutputSpec> outputs;
  outputs.emplace_back("MC", "MCHEADER", 0, Lifetime::Timeframe);
  outputs.emplace_back("MC", "MCTRACKS", 0, Lifetime::Timeframe);
  DataProcessorSpec dSpec = adaptAnalysisTask<KinePublisher>(cfgc, TaskName{"o2sim-kine-publisher"});
  dSpec.outputs = outputs;
  specs.emplace_back(dSpec);
  return specs;
}