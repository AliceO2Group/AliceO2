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
#include "SimulationDataFormat/MCEventHeader.h"
#include "SimulationDataFormat/MCUtils.h"
#include "Framework/AnalysisDataModel.h"
#include "SimulationDataFormat/InteractionSampler.h"

using namespace o2::framework;

struct MctracksToAod {
  Produces<o2::aod::McCollisions> mcollisions;
  Produces<o2::aod::StoredMcParticles_001> mcparticles;
  Configurable<float> IR{"interaction-rate", 100.f, "Interaction rate to simulate"};

  uint64_t timeframe = 0;
  o2::steer::InteractionSampler sampler;

  void init(o2::framework::InitContext& /*ic*/)
  {
    sampler.setInteractionRate(IR);
    sampler.setFirstIR({0, 0});
    sampler.init();
  }

  void run(o2::framework::ProcessingContext& pc)
  {
    auto Nparts = pc.inputs().getNofParts(0);
    auto Nparts_verify = pc.inputs().getNofParts(1);
    if (Nparts != Nparts_verify) {
      LOG(warn) << "Mismatch between number of MC headers and number of track vectors: " << Nparts << " != " << Nparts_verify << ", shipping the empty timeframe";
      return;
    }
    // TODO: include BC simulation
    auto bcCounter = 0UL;
    for (auto i = 0U; i < Nparts; ++i) {
      auto record = sampler.generateCollisionTime();
      auto mcheader = pc.inputs().get<o2::dataformats::MCEventHeader*>("mcheader", i);
      auto mctracks = pc.inputs().get<std::vector<o2::MCTrack>>("mctracks", i);

      mcollisions(bcCounter++, // bcId
                  0,           // generatorId
                  mcheader->GetX(),
                  mcheader->GetY(),
                  mcheader->GetZ(),
                  record.timeInBCNS * 1.e-3, // ns to ms
                  1.,                        // weight
                  mcheader->GetB());
      for (auto& mctrack : mctracks) {
        std::vector<int> mothers;
        int daughters[2];

        if (mctrack.getMotherTrackId() >= 0) {
          mothers.push_back(mctrack.getMotherTrackId());
        }
        if (mctrack.getSecondMotherTrackId() >= 0) {
          mothers.push_back(mctrack.getSecondMotherTrackId());
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
        int statusCode = 0;
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
        if (!mctrack.isPrimary()) {
          flags |= o2::aod::mcparticle::enums::ProducedByTransport; // mark as produced by transport
          statusCode = mctrack.getProcess();
        } else {
          statusCode = mctrack.getStatusCode().fullEncoding;
        }
        if (o2::mcutils::MCTrackNavigator::isPhysicalPrimary(mctrack, mctracks)) {
          flags |= o2::aod::mcparticle::enums::PhysicalPrimary; // mark as physical primary
        }
        mcparticles(mcollisions.lastIndex(), // collisionId,
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
    }
    ++timeframe;
    pc.outputs().snapshot(Output{"TFF", "TFFilename", 0, Lifetime::Timeframe}, "");
    pc.outputs().snapshot(Output{"TFN", "TFNumber", 0, Lifetime::Timeframe}, timeframe);
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  auto dSpec = adaptAnalysisTask<MctracksToAod>(cfgc);
  dSpec.inputs.emplace_back("mctracks", "MC", "MCTRACKS", 0., Lifetime::Timeframe);
  dSpec.inputs.emplace_back("mcheader", "MC", "MCHEADER", 0., Lifetime::Timeframe);
  dSpec.outputs.emplace_back("TFF", "TFFilename");
  dSpec.outputs.emplace_back("TFN", "TFNumber");

  return {dSpec};
}
