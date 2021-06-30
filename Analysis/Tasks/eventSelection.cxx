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

#include "Framework/ConfigParamSpec.h"
using namespace o2;
using namespace o2::framework;

// custom configurable for switching between run2 and run3 selection types
void customize(std::vector<ConfigParamSpec>& workflowOptions)
{
  workflowOptions.push_back(ConfigParamSpec{"selection-run", VariantType::Int, 2, {"selection type: 2 - run 2, 3 - run 3"}});
}

#include "Framework/runDataProcessing.h"
#include "Framework/AnalysisTask.h"
#include "Framework/AnalysisDataModel.h"
#include "AnalysisDataModel/EventSelection.h"
#include "AnalysisCore/TriggerAliases.h"
#include <CCDB/BasicCCDBManager.h>
#include "CommonConstants/LHCConstants.h"

using namespace o2::aod;

struct EvSelParameters {
  // time-of-flight offset
  float fV0ADist = 329.00 / TMath::Ccgs() * 1e9; // ns
  float fV0CDist = 87.15 / TMath::Ccgs() * 1e9;  // ns

  float fFDADist = (1695.30 + 1698.04) / TMath::Ccgs() * 1e9; // ns
  float fFDCDist = (1952.90 + 1955.90) / TMath::Ccgs() * 1e9; // ns

  // beam-beam and beam-gas windows
  float fV0ABBlower = +fV0ADist - 9.5;  // ns
  float fV0ABBupper = +fV0ADist + 22.5; // ns
  float fV0ABGlower = -fV0ADist - 2.5;  // ns
  float fV0ABGupper = -fV0ADist + 5.0;  // ns
  float fV0CBBlower = +fV0CDist - 2.5;  // ns
  float fV0CBBupper = +fV0CDist + 22.5; // ns
  float fV0CBGlower = -fV0CDist - 2.5;  // ns
  float fV0CBGupper = -fV0CDist + 2.5;  // ns

  float fFDABBlower = +fFDADist - 2.5; // ns
  float fFDABBupper = +fFDADist + 2.5; // ns
  float fFDABGlower = -fFDADist - 4.0; // ns
  float fFDABGupper = -fFDADist + 4.0; // ns

  float fFDCBBlower = +fFDCDist - 1.5; // ns
  float fFDCBBupper = +fFDCDist + 1.5; // ns
  float fFDCBGlower = -fFDCDist - 2.0; // ns
  float fFDCBGupper = -fFDCDist + 2.0; // ns

  float fZNABBlower = -2.0;  // ns
  float fZNABBupper = 2.0;   // ns
  float fZNCBBlower = -2.0;  // ns
  float fZNCBBupper = 2.0;   // ns
  float fZNABGlower = 5.0;   // ns
  float fZNABGupper = 100.0; // ns
  float fZNCBGlower = 5.0;   // ns
  float fZNCBGupper = 100.0; // ns

  // TODO rough cuts to be adjusted
  float fT0ABBlower = -2.0; // ns
  float fT0ABBupper = 2.0;  // ns
  float fT0CBBlower = -2.0; // ns
  float fT0CBBupper = 2.0;  // ns

  // Default values from AliOADBTriggerAnalysis constructor
  // TODO store adjusted values period-by-period in CCDB
  float fSPDClsVsTklA = 65.f;
  float fSPDClsVsTklB = 4.f;
  float fV0C012vsTklA = 150.f;
  float fV0C012vsTklB = 20.f;
  float fV0MOnVsOfA = -59.56f;
  float fV0MOnVsOfB = 5.22f;
  float fSPDOnVsOfA = -5.62f;
  float fSPDOnVsOfB = 0.85f;
  float fV0CasymA = -25.f;
  float fV0CasymB = 0.15f;

  bool applySelection[kNsel] = {0};
};

struct BcSelectionTask {
  Produces<aod::BcSels> bcsel;
  Service<o2::ccdb::BasicCCDBManager> ccdb;
  EvSelParameters par;
  void init(InitContext&)
  {
    ccdb->setURL("http://ccdb-test.cern.ch:8080");
    ccdb->setCaching(true);
    ccdb->setLocalObjectValidityChecking();
  }

  using BCsWithRun2InfosTimestampsAndMatches = soa::Join<aod::BCs, aod::Run2BCInfos, aod::Timestamps, aod::Run2MatchedToBCSparse>;

  void process(
    BCsWithRun2InfosTimestampsAndMatches::iterator const& bc,
    aod::Zdcs const& zdcs,
    aod::FV0As const& fv0as,
    aod::FV0Cs const& fv0cs,
    aod::FT0s const& ft0s,
    aod::FDDs const& fdds)
  {
    TriggerAliases* aliases = ccdb->getForTimeStamp<TriggerAliases>("Trigger/TriggerAliases", bc.timestamp());
    if (!aliases) {
      LOGF(fatal, "Trigger aliases are not available in CCDB for run=%d at timestamp=%llu", bc.runNumber(), bc.timestamp());
    }

    // fill fired aliases
    int32_t alias[kNaliases] = {0};
    uint64_t triggerMask = bc.triggerMask();
    for (auto& al : aliases->GetAliasToTriggerMaskMap()) {
      alias[al.first] |= (triggerMask & al.second) > 0;
    }
    uint64_t triggerMaskNext50 = bc.triggerMaskNext50();
    for (auto& al : aliases->GetAliasToTriggerMaskNext50Map()) {
      alias[al.first] |= (triggerMaskNext50 & al.second) > 0;
    }

    // get timing info from ZDC, FV0, FT0 and FDD
    float timeZNA = bc.has_zdc() ? bc.zdc().timeZNA() : -999.f;
    float timeZNC = bc.has_zdc() ? bc.zdc().timeZNC() : -999.f;
    float timeV0A = bc.has_fv0a() ? bc.fv0a().time() : -999.f;
    float timeV0C = bc.has_fv0c() ? bc.fv0c().time() : -999.f;
    float timeT0A = bc.has_ft0() ? bc.ft0().timeA() : -999.f;
    float timeT0C = bc.has_ft0() ? bc.ft0().timeC() : -999.f;
    float timeFDA = bc.has_fdd() ? bc.fdd().timeA() : -999.f;
    float timeFDC = bc.has_fdd() ? bc.fdd().timeC() : -999.f;

    LOGF(debug, "timeZNA=%f timeZNC=%f", timeZNA, timeZNC);
    LOGF(debug, "timeV0A=%f timeV0C=%f", timeV0A, timeV0C);
    LOGF(debug, "timeFDA=%f timeFDC=%f", timeFDA, timeFDC);
    LOGF(debug, "timeT0A=%f timeT0C=%f", timeT0A, timeT0C);

    // applying timing selections
    bool bbV0A = timeV0A > par.fV0ABBlower && timeV0A < par.fV0ABBupper;
    bool bbV0C = timeV0C > par.fV0CBBlower && timeV0C < par.fV0CBBupper;
    bool bbFDA = timeFDA > par.fFDABBlower && timeFDA < par.fFDABBupper;
    bool bbFDC = timeFDC > par.fFDCBBlower && timeFDC < par.fFDCBBupper;
    bool bgV0A = timeV0A > par.fV0ABGlower && timeV0A < par.fV0ABGupper;
    bool bgV0C = timeV0C > par.fV0CBGlower && timeV0C < par.fV0CBGupper;
    bool bgFDA = timeFDA > par.fFDABGlower && timeFDA < par.fFDABGupper;
    bool bgFDC = timeFDC > par.fFDCBGlower && timeFDC < par.fFDCBGupper;

    // fill time-based selection criteria
    int32_t selection[kNsel] = {0}; // TODO switch to bool array
    selection[kIsBBV0A] = bbV0A;
    selection[kIsBBV0C] = bbV0C;
    selection[kIsBBFDA] = bgFDA;
    selection[kIsBBFDC] = bgFDC;
    selection[kNoBGFDA] = !bgFDA;
    selection[kNoBGFDC] = !bgFDC;
    selection[kNoBGFDA] = !bgFDA;
    selection[kNoBGFDC] = !bgFDC;
    selection[kIsBBT0A] = timeT0A > par.fT0ABBlower && timeT0A < par.fT0ABBupper;
    selection[kIsBBT0C] = timeT0C > par.fT0CBBlower && timeT0C < par.fT0CBBupper;
    selection[kIsBBZNA] = timeZNA > par.fZNABBlower && timeZNA < par.fZNABBupper;
    selection[kIsBBZNC] = timeZNC > par.fZNCBBlower && timeZNC < par.fZNCBBupper;
    selection[kNoBGZNA] = !(fabs(timeZNA) > par.fZNABGlower && fabs(timeZNA < par.fZNABGupper));
    selection[kNoBGZNC] = !(fabs(timeZNC) > par.fZNCBGlower && fabs(timeZNC < par.fZNCBGupper));

    // Calculate V0 multiplicity per ring
    float multRingV0A[5] = {0.};
    float multRingV0C[4] = {0.};
    float multV0A = 0;
    float multV0C = 0;
    if (bc.has_fv0a()) {
      for (int ring = 0; ring < 4; ring++) { // TODO adjust for Run3 V0A
        for (int sector = 0; sector < 8; sector++) {
          multRingV0A[ring] += bc.fv0a().amplitude()[ring * 8 + sector];
        }
        multV0A += multRingV0A[ring];
      }
    }

    if (bc.has_fv0c()) {
      for (int ring = 0; ring < 4; ring++) {
        for (int sector = 0; sector < 8; sector++) {
          multRingV0C[ring] += bc.fv0c().amplitude()[ring * 8 + sector];
        }
        multV0C += multRingV0C[ring];
      }
    }
    uint32_t spdClusters = bc.spdClustersL0() + bc.spdClustersL1();

    // Calculate pileup and background related selection flags
    float multV0C012 = multRingV0C[0] + multRingV0C[1] + multRingV0C[2];
    float ofV0M = multV0A + multV0C;
    float onV0M = bc.v0TriggerChargeA() + bc.v0TriggerChargeC();
    float ofSPD = bc.spdFiredChipsL0() + bc.spdFiredChipsL1();
    float onSPD = bc.spdFiredFastOrL0() + bc.spdFiredFastOrL1();
    selection[kNoV0MOnVsOfPileup] = onV0M > par.fV0MOnVsOfA + par.fV0MOnVsOfB * ofV0M;
    selection[kNoSPDOnVsOfPileup] = onSPD > par.fSPDOnVsOfA + par.fSPDOnVsOfB * ofSPD;
    selection[kNoV0Casymmetry] = multRingV0C[3] > par.fV0CasymA + par.fV0CasymB * multV0C012;

    // copy remaining selection decisions from eventCuts
    uint32_t eventCuts = bc.eventCuts();
    selection[kIsGoodTimeRange] = (eventCuts & 1 << aod::kTimeRangeCut) > 0;
    selection[kNoIncompleteDAQ] = (eventCuts & 1 << aod::kIncompleteDAQ) > 0;
    selection[kNoTPCLaserWarmUp] = (eventCuts & 1 << aod::kIsTPCLaserWarmUp) == 0;
    selection[kNoTPCHVdip] = (eventCuts & 1 << aod::kIsTPCHVdip) == 0;
    selection[kNoPileupFromSPD] = (eventCuts & 1 << aod::kIsPileupFromSPD) == 0;
    selection[kNoV0PFPileup] = (eventCuts & 1 << aod::kIsV0PFPileup) == 0;

    int64_t foundFT0 = bc.has_ft0() ? bc.ft0().globalIndex() : -1;

    // Fill bc selection columns
    bcsel(alias, selection,
          bbV0A, bbV0C, bgV0A, bgV0C,
          bbFDA, bbFDC, bgFDA, bgFDC,
          multRingV0A, multRingV0C, spdClusters, foundFT0);
  }
};

struct BcSelectionTaskRun3 {
  Produces<aod::BcSels> bcsel;
  EvSelParameters par;
  using BCsWithMatchings = soa::Join<aod::BCs, aod::Run3MatchedToBCSparse>;
  void process(BCsWithMatchings::iterator const& bc,
               aod::Zdcs const& zdcs,
               aod::FV0As const& fv0as,
               aod::FT0s const& ft0s,
               aod::FDDs const& fdds)
  {
    // TODO: fill fired aliases for run3
    int32_t alias[kNaliases] = {0};

    // get timing info from ZDC, FV0, FT0 and FDD
    float timeZNA = bc.has_zdc() ? bc.zdc().timeZNA() : -999.f;
    float timeZNC = bc.has_zdc() ? bc.zdc().timeZNC() : -999.f;
    float timeV0A = bc.has_fv0a() ? bc.fv0a().time() : -999.f;
    float timeT0A = bc.has_ft0() ? bc.ft0().timeA() : -999.f;
    float timeT0C = bc.has_ft0() ? bc.ft0().timeC() : -999.f;
    float timeFDA = bc.has_fdd() ? bc.fdd().timeA() : -999.f;
    float timeFDC = bc.has_fdd() ? bc.fdd().timeC() : -999.f;

    // applying timing selections
    bool bbV0A = timeV0A > par.fV0ABBlower && timeV0A < par.fV0ABBupper;
    bool bbFDA = timeFDA > par.fFDABBlower && timeFDA < par.fFDABBupper;
    bool bbFDC = timeFDC > par.fFDCBBlower && timeFDC < par.fFDCBBupper;
    bool bgV0A = timeV0A > par.fV0ABGlower && timeV0A < par.fV0ABGupper;
    bool bgFDA = timeFDA > par.fFDABGlower && timeFDA < par.fFDABGupper;
    bool bgFDC = timeFDC > par.fFDCBGlower && timeFDC < par.fFDCBGupper;
    bool bbV0C = 0;
    bool bgV0C = 0;

    // fill time-based selection criteria
    int32_t selection[kNsel] = {0}; // TODO switch to bool array
    selection[kIsBBV0A] = bbV0A;
    selection[kIsBBFDA] = bbFDA;
    selection[kIsBBFDC] = bbFDC;
    selection[kNoBGV0A] = !bgV0A;
    selection[kNoBGFDA] = !bgFDA;
    selection[kNoBGFDC] = !bgFDC;
    selection[kIsBBT0A] = timeT0A > par.fT0ABBlower && timeT0A < par.fT0ABBupper;
    selection[kIsBBT0C] = timeT0C > par.fT0CBBlower && timeT0C < par.fT0CBBupper;
    selection[kIsBBZNA] = timeZNA > par.fZNABBlower && timeZNA < par.fZNABBupper;
    selection[kIsBBZNC] = timeZNC > par.fZNCBBlower && timeZNC < par.fZNCBBupper;
    selection[kNoBGZNA] = !(fabs(timeZNA) > par.fZNABGlower && fabs(timeZNA < par.fZNABGupper));
    selection[kNoBGZNC] = !(fabs(timeZNC) > par.fZNCBGlower && fabs(timeZNC < par.fZNCBGupper));

    // Calculate V0 multiplicity per ring
    float multRingV0A[5] = {0.};
    float multRingV0C[4] = {0.};
    if (bc.has_fv0a()) {
      for (int ring = 0; ring < 5; ring++) {
        for (int sector = 0; sector < 8; sector++) {
          multRingV0A[ring] += bc.fv0a().amplitude()[ring * 8 + sector];
        }
      }
    }

    uint32_t spdClusters = 0;

    int64_t foundFT0 = bc.has_ft0() ? bc.ft0().globalIndex() : -1;
    LOGP(INFO, "foundFT0={}\n", foundFT0);
    // Fill bc selection columns
    bcsel(alias, selection,
          bbV0A, bbV0C, bgV0A, bgV0C,
          bbFDA, bbFDC, bgFDA, bgFDC,
          multRingV0A, multRingV0C, spdClusters, foundFT0);
  }
};

using BCsWithBcSels = soa::Join<aod::BCs, aod::BcSels>;

struct EventSelectionTask {
  Produces<aod::EvSels> evsel;
  Configurable<std::string> syst{"syst", "PbPb", "pp, pPb, Pbp, PbPb, XeXe"}; // TODO determine from AOD metadata or from CCDB
  Configurable<int> muonSelection{"muonSelection", 0, "0 - barrel, 1 - muon selection with pileup cuts, 2 - muon selection without pileup cuts"};
  Configurable<bool> isMC{"isMC", 0, "0 - data, 1 - MC"};
  Partition<aod::Tracks> tracklets = (aod::track::trackType == static_cast<uint8_t>(o2::aod::track::TrackTypeEnum::Run2Tracklet));
  EvSelParameters par;

  void init(InitContext&)
  {
    // TODO store selection criteria in CCDB
    TString systStr = syst.value;
    if (systStr == "PbPb" || systStr == "XeXe") {
      par.applySelection[kIsBBV0A] = 1;
      par.applySelection[kIsBBV0C] = 1;
      par.applySelection[kIsBBZNA] = isMC ? 0 : 1;
      par.applySelection[kIsBBZNC] = isMC ? 0 : 1;
      if (!muonSelection) {
        par.applySelection[kIsGoodTimeRange] = 0; // TODO: not good for run 244918 for some reason - to be checked
        par.applySelection[kNoTPCHVdip] = isMC ? 0 : 1;
      }
    } else if (systStr == "pp") {
      par.applySelection[kIsBBV0A] = 1;
      par.applySelection[kIsBBV0C] = 1;
      par.applySelection[kIsGoodTimeRange] = isMC ? 0 : 1;
      par.applySelection[kNoIncompleteDAQ] = isMC ? 0 : 1;
      par.applySelection[kNoV0C012vsTklBG] = isMC ? 0 : 1;
      par.applySelection[kNoV0Casymmetry] = isMC ? 0 : 1;
      if (muonSelection != 2) {
        par.applySelection[kNoSPDClsVsTklBG] = isMC ? 0 : 1;
        par.applySelection[kNoV0MOnVsOfPileup] = isMC ? 0 : 1;
        par.applySelection[kNoSPDOnVsOfPileup] = isMC ? 0 : 1;
        par.applySelection[kNoPileupFromSPD] = isMC ? 0 : 1;
      }
      if (!muonSelection) {
        par.applySelection[kIsGoodTimeRange] = isMC ? 0 : 1;
        par.applySelection[kNoTPCHVdip] = isMC ? 0 : 1;
      }
    } else if (systStr == "pPb") {
      par.applySelection[kIsBBV0A] = 1;
      par.applySelection[kIsBBV0C] = 1;
      par.applySelection[kNoIncompleteDAQ] = isMC ? 0 : 1;
      par.applySelection[kNoV0C012vsTklBG] = isMC ? 0 : 1;
      par.applySelection[kNoV0Casymmetry] = isMC ? 0 : 1;
      par.applySelection[kNoBGZNA] = isMC ? 0 : 1;
      if (muonSelection != 2) {
        par.applySelection[kNoSPDClsVsTklBG] = isMC ? 0 : 1;
        par.applySelection[kNoV0MOnVsOfPileup] = isMC ? 0 : 1;
        par.applySelection[kNoSPDOnVsOfPileup] = isMC ? 0 : 1;
        par.applySelection[kNoPileupFromSPD] = isMC ? 0 : 1;
      }
      if (!muonSelection) {
        par.applySelection[kIsGoodTimeRange] = isMC ? 0 : 1;
        par.applySelection[kNoTPCHVdip] = isMC ? 0 : 1;
      }
    } else if (systStr == "Pbp") {
      par.applySelection[kIsBBV0A] = 1;
      par.applySelection[kIsBBV0C] = 1;
      par.applySelection[kNoIncompleteDAQ] = isMC ? 0 : 1;
      par.applySelection[kNoV0C012vsTklBG] = isMC ? 0 : 1;
      par.applySelection[kNoV0Casymmetry] = isMC ? 0 : 1;
      par.applySelection[kNoBGZNC] = isMC ? 0 : 1;
      if (muonSelection != 2) {
        par.applySelection[kNoSPDClsVsTklBG] = isMC ? 0 : 1;
        par.applySelection[kNoV0MOnVsOfPileup] = isMC ? 0 : 1;
        par.applySelection[kNoSPDOnVsOfPileup] = isMC ? 0 : 1;
        par.applySelection[kNoPileupFromSPD] = isMC ? 0 : 1;
      }
      if (!muonSelection) {
        par.applySelection[kIsGoodTimeRange] = isMC ? 0 : 1;
        par.applySelection[kNoTPCHVdip] = isMC ? 0 : 1;
      }
    }
  }

  void process(aod::Collision const& col, BCsWithBcSels const& bcs, aod::Tracks const& tracks)
  {
    auto bc = col.bc_as<BCsWithBcSels>();
    int64_t foundFT0 = bc.foundFT0();

    // copy alias decisions from bcsel table
    int32_t alias[kNaliases];
    for (int i = 0; i < kNaliases; i++) {
      alias[i] = bc.alias()[i];
    }

    // copy selection decisions from bcsel table
    int32_t selection[kNsel] = {0};
    for (int i = 0; i < kNsel; i++) {
      selection[i] = bc.selection()[i];
    }

    // copy multiplicity per ring and calculate V0C012 multiplicity
    float multRingV0A[5] = {0.};
    float multRingV0C[4] = {0.};
    for (int i = 0; i < 5; i++) {
      multRingV0A[i] = bc.multRingV0A()[i];
    }
    for (int i = 0; i < 4; i++) {
      multRingV0C[i] = bc.multRingV0C()[i];
    }
    float multV0C012 = bc.multRingV0C()[0] + bc.multRingV0C()[1] + bc.multRingV0C()[2];

    // applying selections depending on the number of tracklets
    int nTkl = tracklets.size();
    uint32_t spdClusters = bc.spdClusters();
    selection[kNoSPDClsVsTklBG] = spdClusters < par.fSPDClsVsTklA + nTkl * par.fSPDClsVsTklB;
    selection[kNoV0C012vsTklBG] = !(nTkl < 6 && multV0C012 > par.fV0C012vsTklA + nTkl * par.fV0C012vsTklB);

    // copy beam-beam and beam-gas flags from bcsel table
    bool bbV0A = bc.bbV0A();
    bool bbV0C = bc.bbV0C();
    bool bgV0A = bc.bgV0A();
    bool bgV0C = bc.bgV0C();
    bool bbFDA = bc.bbFDA();
    bool bbFDC = bc.bbFDC();
    bool bgFDA = bc.bgFDA();
    bool bgFDC = bc.bgFDC();

    // apply int7-like selections
    bool sel7 = 1;
    for (int i = 0; i < kNsel; i++) {
      sel7 &= par.applySelection[i] ? selection[i] : 1;
    }

    // TODO apply other cuts for sel8
    // TODO introduce sel1 etc?
    // TODO introduce array of sel[0]... sel[8] or similar?
    bool sel8 = selection[kIsBBT0A] & selection[kIsBBT0C];

    evsel(alias, selection,
          bbV0A, bbV0C, bgV0A, bgV0C,
          bbFDA, bbFDC, bgFDA, bgFDC,
          multRingV0A, multRingV0C, spdClusters, nTkl, sel7, sel8,
          foundFT0);
  }
};

struct EventSelectionTaskRun3 {
  Produces<aod::EvSels> evsel;

  void process(aod::Collision const& col, BCsWithBcSels const& bcs)
  {
    auto bc = col.bc_as<BCsWithBcSels>();
    int64_t foundFT0 = bc.foundFT0();

    if (foundFT0 < 0) { // search in +/-4 sigma around meanBC
      uint64_t apprBC = bc.globalBC();
      uint64_t meanBC = apprBC - std::lround(col.collisionTime() / o2::constants::lhc::LHCBunchSpacingNS);
      int64_t deltaBC = std::ceil(col.collisionTimeRes() / o2::constants::lhc::LHCBunchSpacingNS * 4);
      // search forward
      int forwardMoveCount = 0;
      int64_t forwardBcDist = deltaBC + 1;
      for (; bc != bcs.end() && bc.globalBC() - meanBC <= deltaBC; ++bc, ++forwardMoveCount) {
        if (bc.foundFT0() >= 0) {
          forwardBcDist = bc.globalBC() - meanBC;
          break;
        }
      }
      bc.moveByIndex(-forwardMoveCount);
      // search backward
      int backwardMoveCount = 0;
      int64_t backwardBcDist = deltaBC + 1;
      for (; bc != bcs.begin() && bc.globalBC() - meanBC >= -deltaBC; --bc, --backwardMoveCount) {
        if (bc.foundFT0() >= 0) {
          backwardBcDist = meanBC - bc.globalBC();
          break;
        }
      }
      if (forwardBcDist > deltaBC && backwardBcDist > deltaBC) {
        bc.moveByIndex(-backwardMoveCount); // return to nominal bc if neighbouring ft0 is not found
      } else if (forwardBcDist < backwardBcDist) {
        bc.moveByIndex(-backwardMoveCount + forwardMoveCount); // move forward
      }
    }
    LOGP(INFO, "{}", bc.foundFT0());

    // copy alias decisions from bcsel table
    int32_t alias[kNaliases];
    for (int i = 0; i < kNaliases; i++) {
      alias[i] = bc.alias()[i];
    }

    // copy selection decisions from bcsel table
    int32_t selection[kNsel] = {0};
    for (int i = 0; i < kNsel; i++) {
      selection[i] = bc.selection()[i];
    }

    // copy multiplicity per ring
    float multRingV0A[5] = {0.};
    float multRingV0C[4] = {0.};
    for (int i = 0; i < 5; i++) {
      multRingV0A[i] = bc.multRingV0A()[i];
    }
    for (int i = 0; i < 4; i++) {
      multRingV0C[i] = bc.multRingV0C()[i];
    }

    int nTkl = 0;
    uint32_t spdClusters = 0;

    // copy beam-beam and beam-gas flags from bcsel table
    bool bbV0A = bc.bbV0A();
    bool bbV0C = bc.bbV0C();
    bool bgV0A = bc.bgV0A();
    bool bgV0C = bc.bgV0C();
    bool bbFDA = bc.bbFDA();
    bool bbFDC = bc.bbFDC();
    bool bgFDA = bc.bgFDA();
    bool bgFDC = bc.bgFDC();

    // apply int7-like selections
    bool sel7 = 0;

    // TODO apply other cuts for sel8
    // TODO introduce sel1 etc?
    // TODO introduce array of sel[0]... sel[8] or similar?
    bool sel8 = selection[kIsBBT0A] & selection[kIsBBT0C];

    evsel(alias, selection,
          bbV0A, bbV0C, bgV0A, bgV0C,
          bbFDA, bbFDC, bgFDA, bgFDC,
          multRingV0A, multRingV0C, spdClusters, nTkl, sel7, sel8,
          foundFT0);
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  if (cfgc.options().get<int>("selection-run") == 2 || cfgc.options().get<int>("selection-run") == 0) {
    return WorkflowSpec{
      adaptAnalysisTask<BcSelectionTask>(cfgc),
      adaptAnalysisTask<EventSelectionTask>(cfgc)};
  } else {
    return WorkflowSpec{
      adaptAnalysisTask<BcSelectionTaskRun3>(cfgc),
      adaptAnalysisTask<EventSelectionTaskRun3>(cfgc)};
  }
}
