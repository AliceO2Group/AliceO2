// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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

  float fZNABBlower = -2.0; // ns
  float fZNABBupper = 2.0;  // ns
  float fZNCBBlower = -2.0; // ns
  float fZNCBBupper = 2.0;  // ns

  // TODO rough cuts to be adjusted
  float fT0ABBlower = -2.0; // ns
  float fT0ABBupper = 2.0;  // ns
  float fT0CBBlower = -2.0; // ns
  float fT0CBBupper = 2.0;  // ns
};

struct EventSelectionTask {
  Produces<aod::EvSels> evsel;
  Service<o2::ccdb::BasicCCDBManager> ccdb;
  Configurable<bool> isMC{"isMC", 0, "0 - data, 1 - MC"};

  EvSelParameters par;

  void init(InitContext&)
  {
    ccdb->setURL("http://ccdb-test.cern.ch:8080");
    ccdb->setCaching(true);
    ccdb->setLocalObjectValidityChecking();
  }

  void process(aod::Run2MatchedSparse::iterator const& collision, aod::BCsWithTimestamps const&, aod::Zdcs const& zdcs, aod::FV0As const& fv0as, aod::FV0Cs const& fv0cs, aod::FT0s const& ft0s, aod::FDDs const& fdds)
  {
    auto bc = collision.bc_as<aod::BCsWithTimestamps>();
    LOGF(debug, "timestamp=%llu", bc.timestamp());
    TriggerAliases* aliases = ccdb->getForTimeStamp<TriggerAliases>("Trigger/TriggerAliases", bc.timestamp());
    if (!aliases) {
      LOGF(fatal, "Trigger aliases are not available in CCDB for run=%d at timestamp=%llu", bc.runNumber(), bc.timestamp());
    }
    uint64_t triggerMask = bc.triggerMask();
    LOGF(debug, "triggerMask=%llu", triggerMask);

    // fill fired aliases
    int32_t alias[kNaliases] = {0};
    for (auto& al : aliases->GetAliasToClassIdsMap()) {
      for (auto& classIndex : al.second) {
        alias[al.first] |= (triggerMask & (1ull << classIndex)) > 0;
      }
    }

    float timeZNA = collision.has_zdc() ? collision.zdc().timeZNA() : -999.f;
    float timeZNC = collision.has_zdc() ? collision.zdc().timeZNC() : -999.f;
    float timeV0A = collision.has_fv0a() ? collision.fv0a().time() : -999.f;
    float timeV0C = collision.has_fv0c() ? collision.fv0c().time() : -999.f;
    float timeT0A = collision.has_ft0() ? collision.ft0().timeA() : -999.f;
    float timeT0C = collision.has_ft0() ? collision.ft0().timeC() : -999.f;
    float timeFDA = collision.has_fdd() ? collision.fdd().timeA() : -999.f;
    float timeFDC = collision.has_fdd() ? collision.fdd().timeC() : -999.f;

    LOGF(debug, "timeZNA=%f timeZNC=%f", timeZNA, timeZNC);
    LOGF(debug, "timeV0A=%f timeV0C=%f", timeV0A, timeV0C);
    LOGF(debug, "timeFDA=%f timeFDC=%f", timeFDA, timeFDC);
    LOGF(debug, "timeT0A=%f timeT0C=%f", timeT0A, timeT0C);

    bool bbZNA = timeZNA > par.fZNABBlower && timeZNA < par.fZNABBupper;
    bool bbZNC = timeZNC > par.fZNCBBlower && timeZNC < par.fZNCBBupper;
    bool bbV0A = timeV0A > par.fV0ABBlower && timeV0A < par.fV0ABBupper;
    bool bbV0C = timeV0C > par.fV0CBBlower && timeV0C < par.fV0CBBupper;
    bool bgV0A = timeV0A > par.fV0ABGlower && timeV0A < par.fV0ABGupper;
    bool bgV0C = timeV0C > par.fV0CBGlower && timeV0C < par.fV0CBGupper;
    bool bbFDA = timeFDA > par.fFDABBlower && timeFDA < par.fFDABBupper;
    bool bbFDC = timeFDC > par.fFDCBBlower && timeFDC < par.fFDCBBupper;
    bool bgFDA = timeFDA > par.fFDABGlower && timeFDA < par.fFDABGupper;
    bool bgFDC = timeFDC > par.fFDCBGlower && timeFDC < par.fFDCBGupper;
    bool bbT0A = timeT0A > par.fT0ABBlower && timeT0A < par.fT0ABBupper;
    bool bbT0C = timeT0C > par.fT0CBBlower && timeT0C < par.fT0CBBupper;

    if (isMC) {
      bbZNA = 1;
      bbZNC = 1;
    }

    // Fill event selection columns
    int64_t foundFT0 = -1; // this column is not used in run2 analysis
    evsel(alias, bbT0A, bbT0C, bbV0A, bbV0C, bgV0A, bgV0C, bbZNA, bbZNC, bbFDA, bbFDC, bgFDA, bgFDC, foundFT0);
  }
};

struct EventSelectionTaskRun3 {
  Produces<aod::EvSels> evsel;

  EvSelParameters par;

  void process(aod::Collision const& collision, soa::Join<aod::BCs, aod::Run3MatchedToBCSparse> const& bct0s, aod::Zdcs const& zdcs, aod::FV0As const& fv0as, aod::FT0s const& ft0s, aod::FDDs const& fdds)
  {
    int64_t ft0Dist;
    int64_t foundFT0 = -1;
    float timeA = -999.f;
    float timeC = -999.f;

    auto bcIter = collision.bc_as<soa::Join<aod::BCs, aod::Run3MatchedToBCSparse>>();

    uint64_t apprBC = bcIter.globalBC();
    uint64_t meanBC = apprBC - std::lround(collision.collisionTime() / o2::constants::lhc::LHCBunchSpacingNS);
    int deltaBC = std::ceil(collision.collisionTimeRes() / o2::constants::lhc::LHCBunchSpacingNS * 4);

    int moveCount = 0;
    while (bcIter != bct0s.end() && bcIter.globalBC() <= meanBC + deltaBC && bcIter.globalBC() >= meanBC - deltaBC) {
      if (bcIter.has_ft0()) {
        ft0Dist = bcIter.globalBC() - meanBC;
        foundFT0 = bcIter.ft0().globalIndex();
        break;
      }
      ++bcIter;
      ++moveCount;
    }

    bcIter.moveByIndex(-moveCount);
    while (bcIter != bct0s.begin() && bcIter.globalBC() <= meanBC + deltaBC && bcIter.globalBC() >= meanBC - deltaBC) {
      --bcIter;
      if (bcIter.has_ft0() && (meanBC - bcIter.globalBC()) < ft0Dist) {
        foundFT0 = bcIter.ft0().globalIndex();
        break;
      }
      if ((meanBC - bcIter.globalBC()) >= ft0Dist) {
        break;
      }
    }

    if (foundFT0 != -1) {
      auto ft0 = ft0s.iteratorAt(foundFT0);
      timeA = ft0.timeA();
      timeC = ft0.timeC();
    }

    bool bbZNA = 1;
    bool bbZNC = 1;
    bool bbV0A = 0;
    bool bbV0C = 0;
    bool bgV0A = 0;
    bool bgV0C = 0;
    bool bbFDA = 0;
    bool bbFDC = 0;
    bool bgFDA = 0;
    bool bgFDC = 0;
    bool bbT0A = timeA > par.fT0ABBlower && timeA < par.fT0ABBupper;
    bool bbT0C = timeC > par.fT0CBBlower && timeC < par.fT0CBBupper;

    int32_t alias[kNaliases] = {0};
    // Fill event selection columns
    // saving FT0 row index (foundFT0) for further analysis
    evsel(alias, bbT0A, bbT0C, bbV0A, bbV0C, bgV0A, bgV0C, bbZNA, bbZNC, bbFDA, bbFDC, bgFDA, bgFDC, foundFT0);
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  if (cfgc.options().get<int>("selection-run") == 2) {
    return WorkflowSpec{adaptAnalysisTask<EventSelectionTask>(cfgc, TaskName{"event-selection"})};
  } else {
    return WorkflowSpec{
      adaptAnalysisTask<EventSelectionTaskRun3>(cfgc, TaskName{"event-selection"})};
  }
}
