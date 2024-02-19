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
#include "Framework/ControlService.h"
#include "Framework/DataTypes.h"
#include "SimulationDataFormat/MCUtils.h"
#include "O2Version.h"
#include "TString.h"

using namespace o2::framework;

namespace o2::aodmcproducer
{

//------------------------------------------------------------------
void AODMcProducerWorkflowDPL::init(InitContext& ic)
{
  mTimer.Stop();
  mLPMProdTag = ic.options().get<std::string>("lpmp-prod-tag");
  mAnchorPass = ic.options().get<std::string>("anchor-pass");
  mAnchorProd = ic.options().get<std::string>("anchor-prod");
  mRecoPass = ic.options().get<std::string>("reco-pass");
  mTFNumber = ic.options().get<int64_t>("aod-timeframe-id");
  mFilterMC = ic.options().get<int>("filter-mctracks");
  int truncate = ic.options().get<int>("enable-truncation");
  if (mTFNumber == -1L) {
    LOG(info) << "TFNumber will be obtained from CCDB";
  }

  // set no truncation if selected by user
  if (truncate == 0) {
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
  std::string hepmcUpdate = ic.options().get<std::string>("hepmc-update");
  HepMCUpdate when = (hepmcUpdate == "never"           //
                        ? HepMCUpdate::never           //
                        : hepmcUpdate == "always"      //
                            ? HepMCUpdate::always      //
                            : hepmcUpdate == "all"     //
                                ? HepMCUpdate::allKeys //
                                : HepMCUpdate::anyKey);
  mXSectionUpdate = when;
  mPdfInfoUpdate = when;
  mHeavyIonUpdate = when;

  mTimer.Reset();
}

//------------------------------------------------------------------
void AODMcProducerWorkflowDPL::updateHeader(CollisionCursor& collisionCursor,
                                            XSectionCursor& xSectionCursor,
                                            PdfInfoCursor& pdfInfoCursor,
                                            HeavyIonCursor& heavyIonCursor,
                                            const MCEventHeader& header,
                                            int collisionID, // Index
                                            int bcID,
                                            float time,
                                            short generatorID,
                                            int sourceID)
{
  using aodmchelpers::updateHepMCHeavyIon;
  using aodmchelpers::updateHepMCPdfInfo;
  using aodmchelpers::updateHepMCXSection;
  using aodmchelpers::updateMCCollisions;

  auto genID = updateMCCollisions(collisionCursor,
                                  bcID,
                                  time,
                                  header,
                                  generatorID,
                                  sourceID,
                                  mCollisionPosition);
  mXSectionUpdate = (updateHepMCXSection(xSectionCursor,  //
                                         collisionID,     //
                                         genID,           //
                                         header,          //
                                         mXSectionUpdate) //
                       ? HepMCUpdate::always              //
                       : HepMCUpdate::never);
  mPdfInfoUpdate = (updateHepMCPdfInfo(pdfInfoCursor,  //
                                       collisionID,    //
                                       genID,          //
                                       header,         //
                                       mPdfInfoUpdate) //
                      ? HepMCUpdate::always            //
                      : HepMCUpdate::never);
  mHeavyIonUpdate = (updateHepMCHeavyIon(heavyIonCursor,  //
                                         collisionID,     //
                                         genID,           //
                                         header,          //
                                         mHeavyIonUpdate) //
                       ? HepMCUpdate::always              //
                       : HepMCUpdate::never);
}

//------------------------------------------------------------------
void AODMcProducerWorkflowDPL::run(ProcessingContext& pc)
{
  mTimer.Start(false);

  uint64_t tfNumber = mTFNumber;

  using namespace o2::aodmchelpers;
  using namespace o2::aodhelpers;

  auto collisionsCursor = createTableCursor<McCollisions>(pc);
  auto particlesCursor = createTableCursor<McParticles>(pc);
  auto originCursor = createTableCursor<Origins>(pc);
  auto xSectionCursor = createTableCursor<XSections>(pc);
  auto pdfInfoCursor = createTableCursor<PdfInfos>(pc);
  auto heavyIonCursor = createTableCursor<HeavyIons>(pc);

  // --- Create our reader -------------------------------------------
  std::unique_ptr<MCKinematicsReader> reader;
  if (not mEnableEmbed) {
    reader =
      std::make_unique<MCKinematicsReader>(mSimPrefix,
                                           MCKinematicsReader::Mode::kMCKine);
  } else {
    reader = std::make_unique<MCKinematicsReader>("collisioncontext.root");
  }

  // --- Container of event indexes ---------------------------------
  using EventInfo = std::vector<std::tuple<int, int, int>>;
  EventInfo eventInfo;

  // --- Fill collision and HepMC aux tables ------------------------
  // dummy time information
  float time = 0;

  if (not mEnableEmbed) {
    // simply store all MC events into table
    int icol = 0;
    int nSources = reader->getNSources();
    for (int isrc = 0; isrc < nSources; isrc++) {
      short generatorID = isrc;
      int nEvents = reader->getNEvents(isrc);
      for (int ievt = 0; ievt < nEvents; ievt++) {
        auto& header = reader->getMCEventHeader(isrc, ievt);
        updateHeader(collisionsCursor.cursor,
                     xSectionCursor.cursor,
                     pdfInfoCursor.cursor,
                     heavyIonCursor.cursor,
                     header,
                     ievt,
                     ievt, // BC is the same as collision index
                     time,
                     generatorID,
                     isrc);

        eventInfo.emplace_back(std::make_tuple(icol, isrc, ievt));
        icol++;
      }
    }
  } else {
    // treat embedded events using collisioncontext: injected events
    // will be stored together with background events into the same
    // collisions
    int nCollisions = reader->getDigitizationContext()->getNCollisions();
    const auto& records = reader->getDigitizationContext()->getEventRecords();
    const auto& parts = reader->getDigitizationContext()->getEventParts();
    for (int icol = 0; icol < nCollisions; icol++) {
      auto& colParts = parts[icol];
      auto nParts = colParts.size();
      for (auto colPart : colParts) {
        auto eventID = colPart.entryID;
        auto sourceID = colPart.sourceID;
        // enable embedding: if several colParts exist, then they are
        // saved as one collision
        if (nParts == 1 || sourceID == 0) {
          // Make collision header from first source only
          short generatorID = sourceID;
          auto& header = reader->getMCEventHeader(sourceID, eventID);

          updateHeader(collisionsCursor.cursor,
                       xSectionCursor.cursor,
                       pdfInfoCursor.cursor,
                       heavyIonCursor.cursor,
                       header,
                       icol,
                       icol, // BC is the same as collision index
                       time,
                       generatorID,
                       sourceID);
        }
        // point background and injected signal events to one collision
        eventInfo.emplace_back(std::make_tuple(icol, sourceID, eventID));
      }
    }
  }

  // Sort the event information
  std::sort(eventInfo.begin(), eventInfo.end(),
            [](typename EventInfo::const_reference left,
               typename EventInfo::const_reference right) { //
              return (std::get<0>(left) < std::get<0>(right));
            });

  // Loop over available events and update the tracks table
  size_t offset = 0;
  for (auto& colInfo : eventInfo) {
    int event = std::get<2>(colInfo);
    int source = std::get<1>(colInfo);
    int collisionID = std::get<0>(colInfo);
    auto tracks = reader->getTracks(source, event);

    TrackToIndex preselect;
    offset = updateParticles(particlesCursor.cursor,
                             collisionID,
                             tracks,
                             preselect,
                             offset,
                             mFilterMC,
                             source == 0,
                             mMcParticleW,
                             mMcParticleMom,
                             mMcParticlePos);

    reader->releaseTracksForSourceAndEvent(source, event);
  }

  // --- Update the origin and the time-frame ------------------------
  originCursor(tfNumber);

  // --- Sending metadata to writer if not done already --------------
  if (not mIsMDSent) {
    TString o2Version = o2::fullVersion();
    std::vector<TString> metaDataKeys = {"DataType",
                                         "Run",
                                         "O2Version",
                                         "ROOTVersion",
                                         "RecoPassName",
                                         "AnchorProduction",
                                         "AnchorPassName",
                                         "LPMProductionTag"};
    std::vector<TString> metaDataVals = {"MC",
                                         "3",
                                         TString{o2::fullVersion()},
                                         ROOT_RELEASE,
                                         mRecoPass,
                                         mAnchorProd,
                                         mAnchorPass,
                                         mLPMProdTag};
    pc.outputs().snapshot(Output{"AMD", "AODMetadataKeys", 0}, metaDataKeys);
    pc.outputs().snapshot(Output{"AMD", "AODMetadataVals", 0}, metaDataVals);
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
  using McCollisions = AODMcProducerWorkflowDPL::McCollisions;
  using McParticles = AODMcProducerWorkflowDPL::McParticles;
  using Origins = AODMcProducerWorkflowDPL::Origins;
  using XSections = AODMcProducerWorkflowDPL::XSections;
  using PdfInfos = AODMcProducerWorkflowDPL::PdfInfos;
  using HeavyIons = AODMcProducerWorkflowDPL::HeavyIons;

  std::vector<OutputSpec> outputs{
    OutputForTable<McCollisions>::spec(),
    OutputForTable<McParticles>::spec(),
    OutputForTable<Origins>::spec(),
    OutputForTable<XSections>::spec(),
    OutputForTable<PdfInfos>::spec(),
    OutputForTable<HeavyIons>::spec(),
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
      ConfigParamSpec{"mckine-fname", VariantType::String, "o2sim", {"MC kinematics file name prefix: e.g. 'o2sim', 'bkg', 'sgn_1'. Used only if 'enable-embedding' is 0"}},
      ConfigParamSpec{"hepmc-update", VariantType::String, "always", {"When to update HepMC Aux tables: always - force update, never - never update, all - if all keys are present, any - when any key is present (not valid yet)"}}}};
}

} // namespace o2::aodmcproducer
//
// EOF
//
