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

// #include "O2RivetExporter.h"
#include "../Detectors/AOD/include/AODProducerWorkflow/AODMcProducerHelpers.h"
#include <Framework/AnalysisTask.h>
#include <SimulationDataFormat/InteractionSampler.h>
#include <Framework/runDataProcessing.h>

template <typename T>
using Configurable = o2::framework::Configurable<T>;

struct MctracksToAod {
  /** @{
      @name Types used */
  using Collisions = o2::aod::McCollisions;
  // using Particles          = o2::aod::McParticles; //
  using Particles = o2::aod::StoredMcParticles_001;
  using XSections = o2::aod::HepMCXSections;
  using PdfInfos = o2::aod::HepMCPdfInfos;
  using HeavyIons = o2::aod::HepMCHeavyIons;
  using InteractionSampler = o2::steer::InteractionSampler;
  /** @} */

  /** @{
      @name Produced data */
  /** Collision header */
  o2::framework::Produces<Collisions> mCollisions;
  /** Particles in collision */
  o2::framework::Produces<Particles> mParticles;
  /** Cross-section information */
  o2::framework::Produces<XSections> mXSections;
  /** Parton-distribution-function information */
  o2::framework::Produces<PdfInfos> mPdfInfos;
  /** Heavy-ion colllision information */
  o2::framework::Produces<HeavyIons> mHeavyIons;
  /** @} */
  /** @{
    @name Configurable parameters */
  Configurable<float> IR{"interaction-rate", 100.f,
                         "Interaction rate to simulate"};
  Configurable<bool> filt{"filter-mctracks", false,
                          "Filter tracks"};
  Configurable<uint64_t> tfOffset{"tf-offset", 0, "Start TF counter from an offset"};
  /** @} */

  /** Number of timeframes */
  uint64_t mTimeFrame = 0;
  /** Interaction simulation */
  InteractionSampler mSampler;

  /** Initialize */
  void init(o2::framework::InitContext& /*ic*/)
  {
    mSampler.setInteractionRate(IR);
    mSampler.setFirstIR({0, 0});
    mSampler.init();

    mTimeFrame = tfOffset;
  }

  /** Run the conversion */
  void run(o2::framework::ProcessingContext& pc)
  {
    LOG(debug) << "=== Running extended MC AOD exporter ===";
    using namespace o2::aodmchelpers;
    using McHeader = o2::dataformats::MCEventHeader;
    using McTrack = o2::MCTrack;
    using McTracks = std::vector<McTrack>;

    auto nParts = pc.inputs().getNofParts(0);
    auto nPartsVerify = pc.inputs().getNofParts(1);

    using o2::framework::Lifetime;
    using o2::framework::Output;

    if (nParts != nPartsVerify) {
      LOG(warn) << "Mismatch between number of MC headers and "
                << "number of track vectors: " << nParts
                << " != " << nPartsVerify
                << ", shipping the empty timeframe";
      pc.outputs().snapshot(Output{"TFF", "TFFilename", 0}, "");
      pc.outputs().snapshot(Output{"TFN", "TFNumber", 0}, ++mTimeFrame);
      return;
    }
    // TODO: include BC simulation
    auto bcCounter = 0UL;
    size_t offset = 0;
    LOG(debug) << "--- Loop over " << nParts << " parts ---";
    for (auto i = 0U; i < nParts; ++i) {
      auto record = mSampler.generateCollisionTime();
      auto header = pc.inputs().get<McHeader*>("mcheader", i);
      auto tracks = pc.inputs().get<McTracks>("mctracks", i);

      LOG(debug) << "Updating collision table";
      auto genID = updateMCCollisions(mCollisions.cursor,
                                      bcCounter,
                                      record.timeInBCNS * 1.e-3,
                                      *header,
                                      0,
                                      i);

      LOG(debug) << "Updating HepMC tables";
      updateHepMCXSection(mXSections.cursor, bcCounter, genID, *header);
      updateHepMCPdfInfo(mPdfInfos.cursor, bcCounter, genID, *header);
      updateHepMCHeavyIon(mHeavyIons.cursor, bcCounter, genID, *header);

      LOG(debug) << "Updating particles table";
      TrackToIndex preselect;
      offset = updateParticles(mParticles.cursor,
                               bcCounter,
                               tracks,
                               preselect,
                               offset,
                               (bool)filt,
                               false);

      LOG(debug) << "Increment BC counter";
      bcCounter++;
    }

    pc.outputs().snapshot(Output{"TFF", "TFFilename", 0}, "");
    pc.outputs().snapshot(Output{"TFN", "TFNumber", 0}, ++mTimeFrame);
  }
};

using WorkflowSpec = o2::framework::WorkflowSpec;
using TaskName = o2::framework::TaskName;
using DataProcessorSpec = o2::framework::DataProcessorSpec;
using ConfigContext = o2::framework::ConfigContext;
using InputSpec = o2::framework::InputSpec;
using OutputSpec = o2::framework::OutputSpec;
using Lifetime = o2::framework::Lifetime;

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  using o2::framework::adaptAnalysisTask;

  auto spec = adaptAnalysisTask<MctracksToAod>(cfgc);
  spec.inputs.emplace_back("mctracks", "MC", "MCTRACKS", 0.,
                           Lifetime::Timeframe);
  spec.inputs.emplace_back("mcheader", "MC", "MCHEADER", 0.,
                           Lifetime::Timeframe);
  spec.outputs.emplace_back("TFF", "TFFilename");
  spec.outputs.emplace_back("TFN", "TFNumber");

  return {spec};
}
//
// EOF
//
