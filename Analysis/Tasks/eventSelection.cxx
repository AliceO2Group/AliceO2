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
#include "CCDB/CcdbApi.h"
#include "TFile.h"
#include "TTree.h"
#include <map>

using std::map;
using std::string;

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

struct EventSelectionTask {
  Produces<aod::EvSels> evsel;
  map<string, int>* pClassNameToIndexMap;
  map<int, string> mAliases;
  map<int, vector<int>> mAliasToClassIds;
  EvSelParameters par;

  aod::Run2V0 getVZero(aod::BC const& bc, aod::Run2V0s const& vzeros)
  {
    for (auto& vzero : vzeros)
      if (vzero.bc() == bc)
        return vzero;
    aod::Run2V0 dummy;
    return dummy;
  }

  aod::Zdc getZdc(aod::BC const& bc, aod::Zdcs const& zdcs)
  {
    for (auto& zdc : zdcs)
      if (zdc.bc() == bc)
        return zdc;
    aod::Zdc dummy;
    return dummy;
  }

  aod::FDD getFDD(aod::BC const& bc, aod::FDDs const& fdds)
  {
    for (auto& fdd : fdds)
      if (fdd.bc() == bc)
        return fdd;
    aod::FDD dummy;
    return dummy;
  }

  // TODO create aliases elsewhere (like BrowseAndFill macro)
  void createAliases()
  {
    mAliases[kINT7] = "CINT7-B-NOPF-CENT,CINT7-B-NOPF-CENTNOTRD";
    mAliases[kEMC7] = "CEMC7-B-NOPF-CENTNOPMD,CDMC7-B-NOPF-CENTNOPMD";
    mAliases[kINT7inMUON] = "CINT7-B-NOPF-MUFAST";
    mAliases[kMuonSingleLowPt7] = "CMSL7-B-NOPF-MUFAST";
    mAliases[kMuonUnlikeLowPt7] = "CMUL7-B-NOPF-MUFAST";
    mAliases[kMuonLikeLowPt7] = "CMLL7-B-NOPF-MUFAST";
    mAliases[kCUP8] = "CCUP8-B-NOPF-CENTNOTRD";
    mAliases[kCUP9] = "CCUP9-B-NOPF-CENTNOTRD";
    mAliases[kMUP10] = "CMUP10-B-NOPF-MUFAST";
    mAliases[kMUP11] = "CMUP11-B-NOPF-MUFAST";
  }

  void init(InitContext&)
  {
    // TODO read run number from configurables
    int run = 244918;

    // read ClassNameToIndexMap from ccdb
    o2::ccdb::CcdbApi ccdb;
    map<string, string> metadata;
    ccdb.init("http://ccdb-test.cern.ch:8080");
    pClassNameToIndexMap = ccdb.retrieveFromTFileAny<map<string, int>>("Trigger/ClassNameToIndexMap", metadata, run);

    LOGF(debug, "List of trigger classes");
    for (auto& cl : *pClassNameToIndexMap) {
      LOGF(debug, "class %02d %s", cl.second, cl.first);
    }

    // TODO read aliases from CCDB
    createAliases();

    LOGF(debug, "Fill map of alias-to-class-indices");
    for (auto& al : mAliases) {
      LOGF(debug, "alias classes: %s", al.second.data());
      TObjArray* tokens = TString(al.second).Tokenize(",");
      for (int iClasses = 0; iClasses < tokens->GetEntriesFast(); iClasses++) {
        string className = tokens->At(iClasses)->GetName();
        int index = (*pClassNameToIndexMap)[className] - 1;
        if (index < 0 || index > 63)
          continue;
        mAliasToClassIds[al.first].push_back(index);
      }
      delete tokens;
    }

    // TODO Fill EvSelParameters with configurable cuts from CCDB
  }

  void process(aod::Collision const& collision, aod::BCs const& bcs, aod::Zdcs const& zdcs, aod::Run2V0s const& vzeros, aod::FDDs const& fdds)
  {
    LOGF(debug, "Starting new event");
    // CTP info
    uint64_t triggerMask = collision.bc().triggerMask();
    LOGF(debug, "triggerMask=%llu", triggerMask);
    // fill fired aliases
    int32_t alias[nAliases] = {0};
    for (auto& al : mAliasToClassIds) {
      for (auto& classIndex : al.second) {
        alias[al.first] |= (triggerMask & (1ull << classIndex)) > 0;
      }
    }

    // for (int i=0;i<64;i++) printf("%i",(triggerMask & (1ull << i)) > 0); printf("\n");
    // for (int i=0;i<nAliases;i++) printf("%i ",alias[i]); printf("\n");

    // ZDC info
    auto zdc = getZdc(collision.bc(), zdcs);
    float timeZNA = zdc.timeZNA();
    float timeZNC = zdc.timeZNC();
    // VZERO info
    auto vzero = getVZero(collision.bc(), vzeros);
    float timeV0A = vzero.timeA();
    float timeV0C = vzero.timeC();
    // FDD info
    auto fdd = getFDD(collision.bc(), fdds);
    float timeFDA = fdd.timeA();
    float timeFDC = fdd.timeC();

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

    // Fill event selection columns
    evsel(alias, bbV0A, bbV0C, bgV0A, bgV0C, bbZNA, bbZNC, bbFDA, bbFDC, bgFDA, bgFDC);
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const&)
{
  return WorkflowSpec{
    adaptAnalysisTask<EventSelectionTask>("event-selection")};
}
