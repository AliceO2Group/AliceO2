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

struct KineToAOD {

  int eventNumber = 0;

  void init(o2::framework::InitContext& ic) {}

  void run(o2::framework::ProcessingContext& pc)
  {

    auto mctracks = pc.inputs().get<std::vector<o2::MCTrack>>("mctracks");
    auto mcheader = pc.inputs().get<o2::dataformats::MCEventHeader*>("mcheader");

    auto& mcCollisionsBuilder = pc.outputs().make<TableBuilder>(Output{"AOD", "MCCOLLISION"});
    auto& mcParticlesBuilder = pc.outputs().make<TableBuilder>(Output{"AOD", "MCPARTICLE_001"});
    auto& mcParticlesEBuilder = pc.outputs().make<TableBuilder>(Output{"AOD", "MCPARTICLE_001E"});

    auto mcCollisionsCursor = mcCollisionsBuilder.cursor<o2::aod::McCollisions>();
    auto mcParticlesCursor = mcParticlesBuilder.cursor<o2::aod::StoredMcParticles_001>();
    auto mcParticlesECursor = mcParticlesEBuilder.cursor<o2::aod::McParticles_001>();

    mcCollisionsCursor(0,
                       0, // bc
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
      int collisionId = 0; // or eventNumber?
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
      mcParticlesCursor(0,
                        collisionId,
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
      float phi = PI + atan2(-1.0f * py, -1.0f * px);
      float p = sqrt(px * px + py * py + pz * pz);
      float pt = sqrt(px * px + py * py);
      float eta;
      if (p - pz < static_cast<float>(1e-7)) {
        if (pz < 0.f) {
          eta = -100.f;
        } else {
          eta = 100.f;
        }
      } else {
        eta = 0.5f * log((p + pz) / (p - pz));
      }
      float Y;
      if (e - pz < static_cast<float>(1e-7)) {
        if (pz < 0.f) {
          Y = -100.f;
        } else {
          Y = 100.f;
        }
      } else {
        Y = 0.5f * log((e + pz) / (e - pz));
      }
      mcParticlesECursor(0,
                         phi,
                         eta,
                         pt,
                         p,
                         Y,
                         collisionId,
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
    eventNumber++;
    pc.outputs().snapshot(Output{"TFF", "TFFilename", 0, Lifetime::Timeframe}, "");
    // pc.outputs().snapshot(Output{"TFN", "TFNumber", 0, Lifetime::Timeframe}, -1L);
    pc.outputs().snapshot(Output{"TFN", "TFNumber", 0, Lifetime::Timeframe}, eventNumber);
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  WorkflowSpec specs;
  std::vector<OutputSpec> outputs;
  outputs.emplace_back(OutputLabel{"O2mccollision"}, "AOD", "MCCOLLISION", 0, Lifetime::Timeframe);
  outputs.emplace_back(OutputLabel{"O2mcparticle_001"}, "AOD", "MCPARTICLE_001", 0, Lifetime::Timeframe);
  outputs.emplace_back(OutputLabel{"O2mcparticle_001E"}, "AOD", "MCPARTICLE_001E", 0, Lifetime::Timeframe);
  outputs.emplace_back(OutputSpec{"TFF", "TFFilename"});
  outputs.emplace_back(OutputSpec{"TFN", "TFNumber"});
  std::vector<InputSpec> inputs;
  inputs.emplace_back("mctracks", "MC", "MCTRACKS", 0., Lifetime::Timeframe);
  inputs.emplace_back("mcheader", "MC", "MCHEADER", 0., Lifetime::Timeframe);
  DataProcessorSpec dSpec = DataProcessorSpec{
    "mctracks-to-aod",
    inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<KineToAOD>()},
    {}};
  specs.emplace_back(dSpec);
  return specs;
}