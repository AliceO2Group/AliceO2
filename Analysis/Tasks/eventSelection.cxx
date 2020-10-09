// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#include "Framework/runDataProcessing.h"
#include "Framework/AnalysisTask.h"
#include "Framework/AnalysisDataModel.h"
#include "Analysis/EventSelection.h"
#include "Analysis/TriggerAliases.h"
#include <CCDB/BasicCCDBManager.h>

using namespace o2;
using namespace o2::framework;

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
};

struct EventSelectionTaskIndexed {
  Produces<aod::EvSels> evsel;
  Service<o2::ccdb::BasicCCDBManager> ccdb;
  Configurable<bool> isMC{"isMC", 0, "0 - data, 1 - MC"};

  static constexpr EvSelParameters par;

  void init(InitContext&)
  {
    ccdb->setURL("http://ccdb-test.cern.ch:8080");
    ccdb->setCaching(true);
    ccdb->setLocalObjectValidityChecking();
  }

  void process(aod::Run2MatchedSparse::iterator const& collision, aod::BCsWithTimestamps const&, aod::Zdcs const&, aod::Run2V0s const&, aod::FDDs const&)
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

    // ZDC info
    float timeZNA = -1.f;
    float timeZNC = -1.f;
    if (collision.has_zdc()) {
      auto zdc = collision.zdc();
      timeZNA = zdc.timeZNA();
      timeZNC = zdc.timeZNC();
    }
    // VZERO info
    float timeV0A = -1.f;
    float timeV0C = -1.f;
    if (collision.has_run2v0()) {
      auto vzero = collision.run2v0();
      timeV0A = vzero.timeA();
      timeV0C = vzero.timeC();
    }
    // FDD info
    float timeFDA = -1.f;
    float timeFDC = -1.f;
    if (collision.has_fdd()) {
      auto fdd = collision.fdd();
      timeFDA = fdd.timeA();
      timeFDC = fdd.timeC();
    }

    LOGF(debug, "timeZNA=%f timeZNC=%f", timeZNA, timeZNC);
    LOGF(debug, "timeV0A=%f timeV0C=%f", timeV0A, timeV0C);
    LOGF(debug, "timeFDA=%f timeFDC=%f", timeFDA, timeFDC);

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

    if (isMC) {
      bbZNA = 1;
      bbZNC = 1;
    }

    // Fill event selection columns
    evsel(alias, bbV0A, bbV0C, bgV0A, bgV0C, bbZNA, bbZNC, bbFDA, bbFDC, bgFDA, bgFDC);
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const&)
{
  return WorkflowSpec{
    adaptAnalysisTask<EventSelectionTaskIndexed>("event-selection")};
}
