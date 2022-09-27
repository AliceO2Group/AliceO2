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

/// @file   AODProducerWorkflowSpec.cxx

#include "AODProducerWorkflow/AODMcProducerWorkflowSpec.h"
#include "AODProducerWorkflow/AODProducerHelpers.h"
#include "CommonUtils/NameConf.h"
#include "MathUtils/Utils.h"
#include "CCDB/BasicCCDBManager.h"
#include "CommonDataFormat/InteractionRecord.h"
#include "DataFormatsGlobalTracking/RecoContainer.h"
#include "Framework/AnalysisDataModel.h"
#include "Framework/ControlService.h"
#include "Framework/DataTypes.h"
#include "Framework/InputRecordWalker.h"
#include "Framework/Logger.h"
#include "Framework/TableBuilder.h"
#include "Framework/TableTreeHelpers.h"
#include "Framework/CCDBParamSpec.h"
#include "ReconstructionDataFormats/VtxTrackIndex.h"
#include "SimulationDataFormat/MCTrack.h"
#include "SimulationDataFormat/MCUtils.h"
#include "O2Version.h"
#include "TMath.h"
#include "TString.h"
#include "TObjString.h"
#include <map>
#include <string>
#include <vector>

using namespace o2::framework;
using namespace o2::math_utils::detail;

namespace o2::aodmcproducer
{

void AODMcProducerWorkflowDPL::collectBCs(const std::vector<o2::InteractionTimeRecord>& mcRecords,
                                          std::map<uint64_t, int>& bcsMap)
{
  bcsMap[mStartIR.toLong()] = 1; // store the start of TF

  // collecting non-empty BCs and enumerating them
  for (auto& rec : mcRecords) {
    uint64_t globalBC = rec.toLong();
    bcsMap[globalBC] = 1;
  }

  int bcID = 0;
  for (auto& item : bcsMap) {
    item.second = bcID;
    bcID++;
  }
}

template <typename MCParticlesCursorType>
void AODMcProducerWorkflowDPL::fillMCParticlesTable(o2::steer::MCKinematicsReader& mcReader,
                                                    const MCParticlesCursorType& mcParticlesCursor,
                                                    const std::map<std::pair<int, int>, int>& mcColToEvSrc)
{
  using o2::aodhelpers::Triplet_t;

  int tableIndex = 1;
  for (auto& colInfo : mcColToEvSrc) { // loop over "<eventID, sourceID> <-> combined MC col. ID" key pairs
    int event = colInfo.first.first;
    int source = colInfo.first.second;
    int mcColId = colInfo.second;
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
      for (int particle = 0; particle < mcParticles.size(); particle++) {
        auto mapItem = mToStore.find(Triplet_t(source, event, particle));
        if (mapItem != mToStore.end()) {
          mapItem->second = tableIndex - 1;
          tableIndex++;
        }
      }
    } else {
      // if all mc particles are stored, all mc particles will be enumerated
      for (int particle = 0; particle < mcParticles.size(); particle++) {
        mToStore[Triplet_t(source, event, particle)] = tableIndex - 1;
        tableIndex++;
      }
    }

    // second part: fill survived mc tracks into the AOD table
    for (int particle = 0; particle < mcParticles.size(); particle++) {
      if (mToStore.find(Triplet_t(source, event, particle)) == mToStore.end()) {
        continue;
      }
      int statusCode = 0;
      uint8_t flags = 0;
      if (!mcParticles[particle].isPrimary()) {
        flags |= o2::aod::mcparticle::enums::ProducedByTransport; // mark as produced by transport
        statusCode = mcParticles[particle].getProcess();
      } else {
        statusCode = mcParticles[particle].getStatusCode();
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
  mLPMProdTag = ic.options().get<string>("lpmp-prod-tag");
  mAnchorPass = ic.options().get<string>("anchor-pass");
  mAnchorProd = ic.options().get<string>("anchor-prod");
  mRecoPass = ic.options().get<string>("reco-pass");
  mTFNumber = ic.options().get<int64_t>("aod-timeframe-id");
  mFilterMC = ic.options().get<int>("filter-mctracks");
  mTruncate = ic.options().get<int>("enable-truncation");
  mRunNumber = ic.options().get<int>("run-number");
  if (mTFNumber == -1L) {
    LOG(info) << "TFNumber will be obtained from CCDB";
  }
  if (mRunNumber == -1L) {
    LOG(info) << "The Run number will be obtained from DPL headers";
  }

  // set no truncation if selected by user
  if (mTruncate != 1) {
    LOG(info) << "Truncation is not used!";
    mCollisionPosition = 0xFFFFFFFF;
    mMcParticleW = 0xFFFFFFFF;
    mMcParticlePos = 0xFFFFFFFF;
    mMcParticleMom = 0xFFFFFFFF;
  }

  // writing metadata if it's not yet in AOD file
  // note: `--aod-writer-resmode "UPDATE"` has to be used,
  //       so that metadata is not overwritten
  mResFile += ".root";
  auto* fResFile = TFile::Open(mResFile, "UPDATE");
  if (!fResFile) {
    LOGF(fatal, "Could not open file %s", mResFile);
  }
  if (fResFile->FindObjectAny("metaData")) {
    LOGF(warning, "Metadata: target file %s already has metadata, preserving it", mResFile);
  } else {
    // populating metadata map
    TString dataType = "MC";
    mMetaData.Add(new TObjString("DataType"), new TObjString(dataType));
    mMetaData.Add(new TObjString("Run"), new TObjString("3"));
    TString O2Version = o2::fullVersion();
    TString ROOTVersion = ROOT_RELEASE;
    mMetaData.Add(new TObjString("O2Version"), new TObjString(O2Version));
    mMetaData.Add(new TObjString("ROOTVersion"), new TObjString(ROOTVersion));
    mMetaData.Add(new TObjString("RecoPassName"), new TObjString(mRecoPass));
    mMetaData.Add(new TObjString("AnchorProduction"), new TObjString(mAnchorProd));
    mMetaData.Add(new TObjString("AnchorPassName"), new TObjString(mAnchorPass));
    mMetaData.Add(new TObjString("LPMProductionTag"), new TObjString(mLPMProdTag));
    LOGF(info, "Metadata: writing into %s", mResFile);
    fResFile->WriteObject(&mMetaData, "metaData", "Overwrite");
  }
  fResFile->Close();

  mTimer.Reset();
}

void AODMcProducerWorkflowDPL::run(ProcessingContext& pc)
{
  mTimer.Start(false);
  mStartIR = pc.services().get<o2::framework::TimingInfo>().firstTForbit;

  const auto& tinfo = pc.services().get<o2::framework::TimingInfo>();

  auto& bcBuilder = pc.outputs().make<TableBuilder>(Output{"AOD", "BC"});
  auto& mcCollisionsBuilder = pc.outputs().make<TableBuilder>(Output{"AOD", "MCCOLLISION"});
  auto& mcParticlesBuilder = pc.outputs().make<TableBuilder>(Output{"AOD", "MCPARTICLE_001"});
  auto& originTableBuilder = pc.outputs().make<TableBuilder>(Output{"AOD", "ORIGIN"});

  auto bcCursor = bcBuilder.cursor<o2::aod::BCs>();
  auto mcCollisionsCursor = mcCollisionsBuilder.cursor<o2::aod::McCollisions>();
  auto mcParticlesCursor = mcParticlesBuilder.cursor<o2::aod::StoredMcParticles_001>();
  auto originCursor = originTableBuilder.cursor<o2::aod::Origins>();

  auto mcReader = std::make_unique<o2::steer::MCKinematicsReader>("collisioncontext.root");
  std::map<uint64_t, int> bcsMap;
  collectBCs(mcReader->getDigitizationContext()->getEventRecords(), bcsMap);

  uint64_t tfNumber;
  const int runNumber = (mRunNumber == -1) ? int(tinfo.runNumber) : mRunNumber;
  if (mTFNumber == -1L) {
    // TODO has to use absolute time of TF
    tfNumber = uint64_t(tinfo.firstTForbit) + (uint64_t(tinfo.runNumber) << 32);
  } else {
    tfNumber = mTFNumber;
  }

  // keep track event/source id for each mc-collision
  // using map and not unordered_map to ensure
  // correct ordering when iterating over container elements
  std::map<std::pair<int, int>, int> mcColToEvSrc;

  // TODO: figure out collision weight
  float mcColWeight = 1.;
  // filling mcCollision table
  int nMCCollisions = mcReader->getDigitizationContext()->getNCollisions();
  const auto& mcRecords = mcReader->getDigitizationContext()->getEventRecords();
  const auto& mcParts = mcReader->getDigitizationContext()->getEventParts();
  for (int iCol = 0; iCol < nMCCollisions; iCol++) {
    auto time = mcRecords[iCol].getTimeNS();
    auto globalBC = mcRecords[iCol].toLong();
    auto item = bcsMap.find(globalBC);
    int bcID = -1;
    if (item != bcsMap.end()) {
      bcID = item->second;
    } else {
      LOG(fatal) << "Error: could not find a corresponding BC ID for MC collision; BC = " << globalBC << ", mc collision = " << iCol;
    }
    auto& colParts = mcParts[iCol];
    auto nParts = colParts.size();
    for (auto colPart : colParts) {
      auto eventID = colPart.entryID;
      auto sourceID = colPart.sourceID;
      // enable embedding: if several colParts exist, then they are saved as one collision
      if (nParts == 1 || sourceID == 0) {
        // FIXME:
        // use generators' names for generatorIDs (?)
        short generatorID = sourceID;
        auto& header = mcReader->getMCEventHeader(sourceID, eventID);
        mcCollisionsCursor(0,
                           bcID,
                           generatorID,
                           truncateFloatFraction(header.GetX(), mCollisionPosition),
                           truncateFloatFraction(header.GetY(), mCollisionPosition),
                           truncateFloatFraction(header.GetZ(), mCollisionPosition),
                           truncateFloatFraction(time, mCollisionPosition),
                           truncateFloatFraction(mcColWeight, mCollisionPosition),
                           header.GetB());
      }
      mcColToEvSrc.emplace(std::pair<int, int>(eventID, sourceID), iCol); // point background and injected signal events to one collision
    }
  }

  // filling BC table
  uint64_t triggerMask = 0;
  for (auto& item : bcsMap) {
    uint64_t bc = item.first;
    bcCursor(0,
             runNumber,
             bc,
             triggerMask);
  }

  bcsMap.clear();

  // filling mc particles table
  fillMCParticlesTable(*mcReader,
                       mcParticlesCursor,
                       mcColToEvSrc);
  mcColToEvSrc.clear();
  mToStore.clear();

  originCursor(0, tfNumber);

  pc.outputs().snapshot(Output{"TFN", "TFNumber", 0, Lifetime::Timeframe}, tfNumber);
  pc.outputs().snapshot(Output{"TFF", "TFFilename", 0, Lifetime::Timeframe}, "");

  pc.services().get<ControlService>().endOfStream();
  pc.services().get<ControlService>().readyToQuit(QuitRequest::Me);

  mTimer.Stop();
}

void AODMcProducerWorkflowDPL::endOfStream(EndOfStreamContext& ec)
{
  LOGF(info, "aod producer dpl total timing: Cpu: %.3e Real: %.3e s in %d slots",
       mTimer.CpuTime(), mTimer.RealTime(), mTimer.Counter() - 1);
}

DataProcessorSpec getAODMcProducerWorkflowSpec(std::string resFile)
{
  std::vector<OutputSpec> outputs;

  outputs.emplace_back(OutputLabel{"O2bc"}, "AOD", "BC", 0, Lifetime::Timeframe);
  outputs.emplace_back(OutputLabel{"O2mccollision"}, "AOD", "MCCOLLISION", 0, Lifetime::Timeframe);
  outputs.emplace_back(OutputLabel{"O2mcparticle_001"}, "AOD", "MCPARTICLE_001", 0, Lifetime::Timeframe);
  outputs.emplace_back(OutputLabel{"O2origin"}, "AOD", "ORIGIN", 0, Lifetime::Timeframe);
  outputs.emplace_back(OutputSpec{"TFN", "TFNumber"});
  outputs.emplace_back(OutputSpec{"TFF", "TFFilename"});

  return DataProcessorSpec{
    "aod-mc-producer-workflow",
    Inputs{},
    outputs,
    AlgorithmSpec{adaptFromTask<AODMcProducerWorkflowDPL>(resFile)},
    Options{
      ConfigParamSpec{"run-number", VariantType::Int64, -1L, {"The run-number. If left default we try to get it from DPL header."}},
      ConfigParamSpec{"aod-timeframe-id", VariantType::Int64, -1L, {"Set timeframe number"}},
      ConfigParamSpec{"enable-truncation", VariantType::Int, 1, {"Truncation parameter: 1 -- on, != 1 -- off"}},
      ConfigParamSpec{"lpmp-prod-tag", VariantType::String, "", {"LPMProductionTag"}},
      ConfigParamSpec{"anchor-pass", VariantType::String, "", {"AnchorPassName"}},
      ConfigParamSpec{"anchor-prod", VariantType::String, "", {"AnchorProduction"}},
      ConfigParamSpec{"reco-pass", VariantType::String, "", {"RecoPassName"}},
      ConfigParamSpec{"filter-mctracks", VariantType::Int, 1, {"Store only physical primary MC tracks and their mothers/daughters. 0 -- off, != 0 -- on"}}}};
}

} // namespace o2::aodmcproducer
