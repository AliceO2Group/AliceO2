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
#include "Framework/Task.h"
#include "SimulationDataFormat/MCTrack.h"
#include "SimulationDataFormat/MCEventHeader.h"
#include "Framework/AnalysisDataModel.h"

using namespace o2::framework;

struct McTracksToAODSpawner {
  Spawns<o2::aod::McParticles> mcparticlesExt;

  void init(o2::framework::InitContext& ic) {}

  void run(o2::framework::ProcessingContext& pc) {}
};

struct McTracksToAOD {

  Produces<o2::aod::McCollisions> mcollisions;
  Produces<o2::aod::StoredMcParticles_001> mcparticles;

  Configurable<int> collisionsPerTimeFfame{"collisionsPerTimeframe", 200, "Number of McCollisions per timeframe"};

  int collisionId = 0;
  long timeframe = 0;

  void init(o2::framework::InitContext& ic) {}

  void run(o2::framework::ProcessingContext& pc)
  {
    auto mcheader = pc.inputs().get<o2::dataformats::MCEventHeader*>("mcheader");
    auto mctracks = pc.inputs().get<std::vector<o2::MCTrack>>("mctracks");

    mcollisions(0, // bc
                0, // generatorId
                mcheader->GetX(),
                mcheader->GetY(),
                mcheader->GetZ(),
                mcheader->GetT(),
                1., // weight
                mcheader->GetB());
    for (auto& mctrack : mctracks) {
      int mothers_size = 0;
      std::vector<int> mothers;
      int daughters[2];

      if (mctrack.getMotherTrackId() >= 0) {
        mothers.push_back(mctrack.getMotherTrackId());
        mothers_size++;
      }
      if (mctrack.getSecondMotherTrackId() >= 0) {
        mothers.push_back(mctrack.getSecondMotherTrackId());
        mothers_size++;
      }
      daughters[0] = -1;
      daughters[1] = -1;
      if (mctrack.getFirstDaughterTrackId() >= 0 && mctrack.getLastDaughterTrackId() >= 0) {
        daughters[0] = mctrack.getFirstDaughterTrackId();
        daughters[1] = mctrack.getLastDaughterTrackId();
      } else if (mctrack.getFirstDaughterTrackId() >= 0) {
        daughters[0] = mctrack.getFirstDaughterTrackId();
        daughters[1] = mctrack.getLastDaughterTrackId();
      }
      int PdgCode = mctrack.GetPdgCode();
      int statusCode = mctrack.getStatusCode().fullEncoding;
      float weight = mctrack.getWeight();
      float px = mctrack.Px();
      float py = mctrack.Py();
      float pz = mctrack.Pz();
      float e = mctrack.GetEnergy();
      float x = mctrack.GetStartVertexCoordinatesX();
      float y = mctrack.GetStartVertexCoordinatesY();
      float z = mctrack.GetStartVertexCoordinatesZ();
      float t = mctrack.GetStartVertexCoordinatesT();
      int flags = 0;
      mcparticles(0, // collisionId,
                  PdgCode,
                  statusCode,
                  flags,
                  mothers,
                  daughters,
                  weight,
                  px,
                  py,
                  pz,
                  e,
                  x,
                  y,
                  z,
                  t);
    }
    collisionId++;
    pc.outputs().snapshot(Output{"TFF", "TFFilename", 0, Lifetime::Timeframe}, "");
    pc.outputs().snapshot(Output{"TFN", "TFNumber", 0, Lifetime::Timeframe}, timeframe);
    if (collisionId == collisionsPerTimeFfame) {
      collisionId = 0;
      timeframe++;
    }
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  WorkflowSpec specs;
  std::vector<InputSpec> inputs;
  inputs.emplace_back("mctracks", "MC", "MCTRACKS", 0., Lifetime::Timeframe);
  inputs.emplace_back("mcheader", "MC", "MCHEADER", 0., Lifetime::Timeframe);
  DataProcessorSpec dSpec = adaptAnalysisTask<McTracksToAOD>(cfgc, TaskName{"mctracks-to-aod"});
  dSpec.inputs = inputs;
  dSpec.outputs.emplace_back(OutputSpec{"TFF", "TFFilename"});
  dSpec.outputs.emplace_back(OutputSpec{"TFN", "TFNumber"});
  specs.emplace_back(dSpec);

  DataProcessorSpec dSpec2 = adaptAnalysisTask<McTracksToAODSpawner>(cfgc, TaskName{"mctracks-to-aod-spawner"});
  specs.emplace_back(dSpec2);

  return specs;
}
