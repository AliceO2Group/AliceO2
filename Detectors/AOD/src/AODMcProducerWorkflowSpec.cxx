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

/// @file   AODMcProducerWorkflowSpec.cxx

#include "AODProducerWorkflow/AODMcProducerWorkflowSpec.h"
#include "AODProducerWorkflow/AODProducerHelpers.h"
#include "Framework/AnalysisDataModel.h"
#include "Framework/ControlService.h"
#include "Framework/DataTypes.h"
#include "Framework/TableBuilder.h"
#include "SimulationDataFormat/MCTrack.h"
#include "SimulationDataFormat/MCUtils.h"
#include "O2Version.h"
#include "TString.h"
#include <vector>

using namespace o2::framework;
using namespace o2::math_utils::detail;

namespace o2::aodmcproducer
{

template <typename MCParticlesCursorType>
void AODMcProducerWorkflowDPL::fillMCParticlesTable(o2::steer::MCKinematicsReader& mcReader,
                                                    const MCParticlesCursorType& mcParticlesCursor)
{
  using o2::aodhelpers::Triplet_t;

  int tableIndex = 1;
  for (auto& colInfo : mMCColToEvSrc) { // loop over "<eventID, sourceID> <-> combined MC col. ID" key pairs
    int event = colInfo[2];
    int source = colInfo[1];
    int mcColId = colInfo[0];
    std::vector<MCTrack> const& mcParticles = mcReader.getTracks(source, event);
    // mark tracks to be stored per event
    // loop over stack of MC particles from end to beginning: daughters are stored after mothers
    if (mFilterMC) {
      for (int particle = mcParticles.size() - 1; particle >= 0; particle--) {
        // we store all primary particles == particles put by generator
        if (mcParticles[particle].isPrimary() ||
            o2::mcutils::MCTrackNavigator::isPhysicalPrimary(mcParticles[particle], mcParticles) ||
            o2::mcutils::MCTrackNavigator::isKeepPhysics(mcParticles[particle], mcParticles)) {
          mToStore[Triplet_t(source, event, particle)] = 1;
        } else {
          continue;
        }
        // we store mothers and daughters of particles that are to be saved
        int mother0 = mcParticles[particle].getMotherTrackId();
        if (mother0 != -1) {
          mToStore[Triplet_t(source, event, mother0)] = 1;
        }
        int mother1 = mcParticles[particle].getSecondMotherTrackId();
        if (mother1 != -1) {
          mToStore[Triplet_t(source, event, mother1)] = 1;
        }
        int daughter0 = mcParticles[particle].getFirstDaughterTrackId();
        if (daughter0 != -1) {
          mToStore[Triplet_t(source, event, daughter0)] = 1;
        }
        int daughterL = mcParticles[particle].getLastDaughterTrackId();
        if (daughterL != -1) {
          mToStore[Triplet_t(source, event, daughterL)] = 1;
        }
      }
      // enumerate saved mc particles and their relatives to get mother/daughter relations
      for (auto particle = 0U; particle < mcParticles.size(); ++particle) {
        auto mapItem = mToStore.find(Triplet_t(source, event, particle));
        if (mapItem != mToStore.end()) {
          mapItem->second = tableIndex - 1;
          tableIndex++;
        }
      }
    } else {
      // if all mc particles are stored, all mc particles will be enumerated
      for (auto particle = 0U; particle < mcParticles.size(); ++particle) {
        mToStore[Triplet_t(source, event, particle)] = tableIndex - 1;
        tableIndex++;
      }
    }

    // second part: fill survived mc tracks into the AOD table
    for (auto particle = 0U; particle < mcParticles.size(); ++particle) {
      if (mToStore.find(Triplet_t(source, event, particle)) == mToStore.end()) {
        continue;
      }
      int statusCode = 0;
      uint8_t flags = 0;
      if (!mcParticles[particle].isPrimary()) {
        flags |= o2::aod::mcparticle::enums::ProducedByTransport; // mark as produced by transport
        statusCode = mcParticles[particle].getProcess();
      } else {
        statusCode = mcParticles[particle].getStatusCode().fullEncoding;
      }
      if (source == 0) {
        flags |= o2::aod::mcparticle::enums::FromBackgroundEvent; // mark as particle from background event
      }
      if (o2::mcutils::MCTrackNavigator::isPhysicalPrimary(mcParticles[particle], mcParticles)) {
        flags |= o2::aod::mcparticle::enums::PhysicalPrimary; // mark as physical primary
      }
      float weight = mcParticles[particle].getWeight();
      std::vector<int> mothers;
      int mcMother0 = mcParticles[particle].getMotherTrackId();
      auto item = mToStore.find(Triplet_t(source, event, mcMother0));
      if (item != mToStore.end()) {
        mothers.push_back(item->second);
      }
      int mcMother1 = mcParticles[particle].getSecondMotherTrackId();
      item = mToStore.find(Triplet_t(source, event, mcMother1));
      if (item != mToStore.end()) {
        mothers.push_back(item->second);
      }
      int daughters[2] = {-1, -1}; // slice
      int mcDaughter0 = mcParticles[particle].getFirstDaughterTrackId();
      item = mToStore.find(Triplet_t(source, event, mcDaughter0));
      if (item != mToStore.end()) {
        daughters[0] = item->second;
      }
      int mcDaughterL = mcParticles[particle].getLastDaughterTrackId();
      item = mToStore.find(Triplet_t(source, event, mcDaughterL));
      if (item != mToStore.end()) {
        daughters[1] = item->second;
        if (daughters[0] < 0) {
          LOG(error) << "AOD problematic daughter case observed";
          daughters[0] = daughters[1]; /// Treat the case of first negative label (pruned in the kinematics)
        }
      } else {
        daughters[1] = daughters[0];
      }
      if (daughters[0] > daughters[1]) {
        std::swap(daughters[0], daughters[1]);
      }
      auto pX = (float)mcParticles[particle].Px();
      auto pY = (float)mcParticles[particle].Py();
      auto pZ = (float)mcParticles[particle].Pz();
      auto energy = (float)mcParticles[particle].GetEnergy();
      mcParticlesCursor(0,
                        mcColId,
                        mcParticles[particle].GetPdgCode(),
                        statusCode,
                        flags,
                        mothers,
                        daughters,
                        truncateFloatFraction(weight, mMcParticleW),
                        truncateFloatFraction(pX, mMcParticleMom),
                        truncateFloatFraction(pY, mMcParticleMom),
                        truncateFloatFraction(pZ, mMcParticleMom),
                        truncateFloatFraction(energy, mMcParticleMom),
                        truncateFloatFraction((float)mcParticles[particle].Vx(), mMcParticlePos),
                        truncateFloatFraction((float)mcParticles[particle].Vy(), mMcParticlePos),
                        truncateFloatFraction((float)mcParticles[particle].Vz(), mMcParticlePos),
                        truncateFloatFraction((float)mcParticles[particle].T(), mMcParticlePos));
    }
    mcReader.releaseTracksForSourceAndEvent(source, event);
  }
}

void AODMcProducerWorkflowDPL::init(InitContext& ic)
{
  mTimer.Stop();
  mLPMProdTag = ic.options().get<std::string>("lpmp-prod-tag");
  mAnchorPass = ic.options().get<std::string>("anchor-pass");
  mAnchorProd = ic.options().get<std::string>("anchor-prod");
  mRecoPass = ic.options().get<std::string>("reco-pass");
  mTFNumber = ic.options().get<int64_t>("aod-timeframe-id");
  mFilterMC = ic.options().get<int>("filter-mctracks");
  mTruncate = ic.options().get<int>("enable-truncation");
  if (mTFNumber == -1L) {
    LOG(info) << "TFNumber will be obtained from CCDB";
  }

  // set no truncation if selected by user
  if (mTruncate != 1) {
    LOG(info) << "Truncation is not used!";
    mCollisionPosition = 0xFFFFFFFF;
    mMcParticleW = 0xFFFFFFFF;
    mMcParticlePos = 0xFFFFFFFF;
    mMcParticleMom = 0xFFFFFFFF;
  }

  mEnableEmbed = ic.options().get<bool>("enable-embedding");

  if (!mEnableEmbed) {
    // parse list of sim prefixes into vector
    mSimPrefix = ic.options().get<std::string>("mckine-fname");
  }

  mTimer.Reset();
}

void AODMcProducerWorkflowDPL::run(ProcessingContext& pc)
{
  mTimer.Start(false);

  uint64_t tfNumber = mTFNumber;

  auto mcCollisionsBuilder = pc.outputs().make<TableBuilder>(OutputForTable<aod::McCollisions>::ref());
  auto mcParticlesBuilder = pc.outputs().make<TableBuilder>(OutputForTable<aod::StoredMcParticles>::ref());
  auto originTableBuilder = pc.outputs().make<TableBuilder>(OutputForTable<aod::Origins>::ref());

  auto mcCollisionsCursor = mcCollisionsBuilder->cursor<o2::aod::McCollisions>();
  auto mcParticlesCursor = mcParticlesBuilder->cursor<o2::aod::StoredMcParticles>();
  auto originCursor = originTableBuilder->cursor<o2::aod::Origins>();

  std::unique_ptr<o2::steer::MCKinematicsReader> mcReader;

  if (!mEnableEmbed) {
    mcReader = std::make_unique<o2::steer::MCKinematicsReader>(mSimPrefix, steer::MCKinematicsReader::Mode::kMCKine);
  } else {
    mcReader = std::make_unique<o2::steer::MCKinematicsReader>("collisioncontext.root");
  }

  // filling mcCollision table
  // dummy time information
  int bcID = 0;
  float time = 0;

  auto updateMCCollisions = [this, bcID, time, &mcCollisionsCursor](dataformats::MCEventHeader const& header, short generatorID, int sourceID) {
    bool isValid = false;
    int subGeneratorId{-1};
    if (header.hasInfo(o2::mcgenid::GeneratorProperty::SUBGENERATORID)) {
      subGeneratorId = header.getInfo<int>(o2::mcgenid::GeneratorProperty::SUBGENERATORID, isValid);
    }
    float mcColWeight = 1.;
    if (header.hasInfo("weight")) {
      mcColWeight = header.getInfo<float>("weight", isValid);
    }
    mcCollisionsCursor(0,
                       bcID,
                       o2::mcgenid::getEncodedGenId(header.getInfo<int>(o2::mcgenid::GeneratorProperty::GENERATORID, isValid), sourceID, subGeneratorId),
                       truncateFloatFraction(header.GetX(), mCollisionPosition),
                       truncateFloatFraction(header.GetY(), mCollisionPosition),
                       truncateFloatFraction(header.GetZ(), mCollisionPosition),
                       truncateFloatFraction(time, mCollisionPosition),
                       truncateFloatFraction(mcColWeight, mCollisionPosition),
                       header.GetB());
  };

  if (!mEnableEmbed) { // simply store all MC events into table
    int icol = 0;
    int nSources = mcReader->getNSources();
    for (int isrc = 0; isrc < nSources; isrc++) {
      short generatorID = isrc;
      int nEvents = mcReader->getNEvents(isrc);
      for (int ievt = 0; ievt < nEvents; ievt++) {
        auto& header = mcReader->getMCEventHeader(isrc, ievt);
        updateMCCollisions(header, generatorID, isrc);
        mMCColToEvSrc.emplace_back(std::vector<int>{icol, isrc, ievt});
        icol++;
      }
    }
  } else { // treat embedded events using collisioncontext: injected events will be stored together with background events into the same collisions
    int nMCCollisions = mcReader->getDigitizationContext()->getNCollisions();
    const auto& mcRecords = mcReader->getDigitizationContext()->getEventRecords();
    const auto& mcParts = mcReader->getDigitizationContext()->getEventParts();
    for (int icol = 0; icol < nMCCollisions; icol++) {
      auto& colParts = mcParts[icol];
      auto nParts = colParts.size();
      for (auto colPart : colParts) {
        auto eventID = colPart.entryID;
        auto sourceID = colPart.sourceID;
        // enable embedding: if several colParts exist, then they are saved as one collision
        if (nParts == 1 || sourceID == 0) {
          short generatorID = sourceID;
          auto& header = mcReader->getMCEventHeader(sourceID, eventID);
          updateMCCollisions(header, generatorID, sourceID);
        }
        mMCColToEvSrc.emplace_back(std::vector<int>{icol, sourceID, eventID}); // point background and injected signal events to one collision
      }
    }
  }

  std::sort(mMCColToEvSrc.begin(), mMCColToEvSrc.end(),
            [](const std::vector<int>& left, const std::vector<int>& right) { return (left[0] < right[0]); });

  // filling mc particles table
  fillMCParticlesTable(*mcReader, mcParticlesCursor);

  mMCColToEvSrc.clear();
  mToStore.clear();

  originCursor(0, tfNumber);

  // sending metadata to writer
  if (!mIsMDSent) {
    TString dataType = "MC";
    TString O2Version = o2::fullVersion();
    TString ROOTVersion = ROOT_RELEASE;
    mMetaDataKeys = {"DataType", "Run", "O2Version", "ROOTVersion", "RecoPassName", "AnchorProduction", "AnchorPassName", "LPMProductionTag"};
    mMetaDataVals = {dataType, "3", O2Version, ROOTVersion, mRecoPass, mAnchorProd, mAnchorPass, mLPMProdTag};
    pc.outputs().snapshot(Output{"AMD", "AODMetadataKeys", 0}, mMetaDataKeys);
    pc.outputs().snapshot(Output{"AMD", "AODMetadataVals", 0}, mMetaDataVals);
    mIsMDSent = true;
  }

  pc.outputs().snapshot(Output{"TFN", "TFNumber", 0}, tfNumber);
  pc.outputs().snapshot(Output{"TFF", "TFFilename", 0}, "");

  pc.services().get<ControlService>().endOfStream();
  pc.services().get<ControlService>().readyToQuit(QuitRequest::Me);

  mTimer.Stop();
}

void AODMcProducerWorkflowDPL::endOfStream(EndOfStreamContext&)
{
  LOGF(info, "aod producer dpl total timing: Cpu: %.3e Real: %.3e s in %d slots",
       mTimer.CpuTime(), mTimer.RealTime(), mTimer.Counter() - 1);
}

DataProcessorSpec getAODMcProducerWorkflowSpec()
{
  std::vector<OutputSpec> outputs{
    OutputForTable<aod::McCollisions>::spec(),
    OutputForTable<aod::StoredMcParticles>::spec(),
    OutputForTable<aod::Origins>::spec(),
    OutputSpec{"TFN", "TFNumber"},
    OutputSpec{"TFF", "TFFilename"},
    OutputSpec{"AMD", "AODMetadataKeys"},
    OutputSpec{"AMD", "AODMetadataVals"}};

  return DataProcessorSpec{
    "aod-mc-producer-workflow",
    Inputs{},
    outputs,
    AlgorithmSpec{adaptFromTask<AODMcProducerWorkflowDPL>()},
    Options{
      ConfigParamSpec{"aod-timeframe-id", VariantType::Int64, 1L, {"Set timeframe number"}},
      ConfigParamSpec{"enable-truncation", VariantType::Int, 1, {"Truncation parameter: 1 -- on, != 1 -- off"}},
      ConfigParamSpec{"lpmp-prod-tag", VariantType::String, "", {"LPMProductionTag"}},
      ConfigParamSpec{"anchor-pass", VariantType::String, "", {"AnchorPassName"}},
      ConfigParamSpec{"anchor-prod", VariantType::String, "", {"AnchorProduction"}},
      ConfigParamSpec{"reco-pass", VariantType::String, "", {"RecoPassName"}},
      ConfigParamSpec{"filter-mctracks", VariantType::Int, 1, {"Store only physical primary MC tracks and their mothers/daughters. 0 -- off, != 0 -- on"}},
      ConfigParamSpec{"enable-embedding", VariantType::Int, 0, {"Use collisioncontext.root to process embedded events"}},
      ConfigParamSpec{"mckine-fname", VariantType::String, "o2sim", {"MC kinematics file name prefix: e.g. 'o2sim', 'bkg', 'sgn_1'. Used only if 'enable-embedding' is 0"}}}};
}

} // namespace o2::aodmcproducer
