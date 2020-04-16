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
#include "TFile.h"
#include "TTree.h"
#include "TH1F.h"
#include <map>

using std::map;
using std::string;

using namespace o2;
using namespace o2::framework;

struct EventSelectionTask {
  Produces<aod::EvSels> evsel;
  map<string, int> mClasses;
  map<int, string> mAliases;

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

  // TODO create aliases elsewhere (like BrowseAndFill macro)
  void createAliases()
  {
    mAliases[0] = "CINT7-B-NOPF-CENT";
    mAliases[1] = "CEMC7-B-NOPF-CENTNOPMD,CDMC7-B-NOPF-CENTNOPMD";
  }

  void init(InitContext&)
  {
    // TODO read aliases from CCDB
    createAliases();

    // TODO read triggerClass-index map from CCDB
    TFile f("trigger.root");
    TTree* t = (TTree*)f.Get("trigger");
    map<string, int>* pClasses = &mClasses;
    t->SetBranchAddress("classes", &pClasses);
    t->BuildIndex("run");
    // TODO read run number from configurables
    t->GetEntryWithIndex(244918);
    LOGF(info, "List of trigger classes");
    for (auto& cl : mClasses) {
      LOGF(info, "class %d %s", cl.second, cl.first);
    }
  }

  void process(aod::Collision const& collision, aod::BCs const& bcs, aod::Zdcs const& zdcs, aod::Run2V0s const& vzeros)
  {
    LOGF(info, "Starting new event");
    // CTP info
    uint64_t triggerMask = collision.bc().triggerMask();
    LOGF(info, "triggerMask=%llu", triggerMask);
    // fill fired aliases
    int32_t alias[nAliases] = {0};
    for (auto& al : mAliases) {
      LOGF(info, "alias classes: %s", al.second.data());
      TObjArray* tokens = TString(al.second).Tokenize(",");
      for (int iClasses = 0; iClasses < tokens->GetEntriesFast(); iClasses++) {
        string className = tokens->At(iClasses)->GetName();
        int index = mClasses[className] - 1;
        if (index < 0 || index > 49)
          continue;
        bool isTriggerClassFired = triggerMask & (1ul << index);
        LOGF(info, "class=%s index=%d fired=%d", className.data(), index, isTriggerClassFired);
        alias[al.first] |= isTriggerClassFired;
      }
      delete tokens;
    }

    // ZDC info
    auto zdc = getZdc(collision.bc(), zdcs);
    // TODO replace it with timing checks when time arrays become available
    uint8_t zdcFired = zdc.fired();
    bool bbZNA = zdcFired & 1 << 0;
    bool bbZNC = zdcFired & 1 << 1;

    // VZERO info
    auto vzero = getVZero(collision.bc(), vzeros);
    // TODO use properly calibrated average times
    float timeV0A = 0;
    float timeV0C = 0;
    int nHitsV0A = 0;
    int nHitsV0C = 0;
    for (int i = 0; i < 32; i++) {
      if (vzero.time()[i + 32] > -998) {
        nHitsV0A++;
        timeV0A += vzero.time()[i + 32];
      }
      if (vzero.time()[i] > -998) {
        nHitsV0C++;
        timeV0C += vzero.time()[i];
      }
    }
    timeV0A = nHitsV0A > 0 ? timeV0A / nHitsV0A : -999;
    timeV0C = nHitsV0C > 0 ? timeV0C / nHitsV0C : -999;

    // TODO replace it with configurable cuts from CCDB
    bool bbV0A = timeV0A > 0. && timeV0A < 25.;  // ns
    bool bbV0C = timeV0C > 0. && timeV0C < 25.;  // ns
    bool bgV0A = timeV0A > -25. && timeV0A < 0.; // ns
    bool bgV0C = timeV0C > -25. && timeV0C < 0.; // ns

    // Fill event selection columns
    evsel(alias, bbV0A, bbV0C, bgV0A, bgV0C, bbZNA, bbZNC);
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const&)
{
  return WorkflowSpec{
    adaptAnalysisTask<EventSelectionTask>("event-selection")};
}
