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
#include <map>

using std::map;
using std::string;

using namespace o2;
using namespace o2::framework;

struct EventSelectionTask {
  Produces<aod::EvSels> evsel;
  map<string, int> mClasses;
  map<int, string> mAliases;
  map<int, vector<int>> mAliasToClassIds;

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
    LOGF(debug, "List of trigger classes");
    for (auto& cl : mClasses) {
      LOGF(debug, "class %d %s", cl.second, cl.first);
    }

    LOGF(debug, "Fill map of alias-to-class-indices");
    for (auto& al : mAliases) {
      LOGF(debug, "alias classes: %s", al.second.data());
      TObjArray* tokens = TString(al.second).Tokenize(",");
      for (int iClasses = 0; iClasses < tokens->GetEntriesFast(); iClasses++) {
        string className = tokens->At(iClasses)->GetName();
        int index = mClasses[className] - 1;
        if (index < 0 || index > 49)
          continue;
        mAliasToClassIds[al.first].push_back(index);
      }
      delete tokens;
    }
  }

  void process(aod::Collision const& collision, aod::BCs const& bcs, aod::Zdcs const& zdcs, aod::Run2V0s const& vzeros)
  {
    LOGF(debug, "Starting new event");
    // CTP info
    uint64_t triggerMask = collision.bc().triggerMask();
    LOGF(debug, "triggerMask=%llu", triggerMask);
    // fill fired aliases
    int32_t alias[nAliases] = {0};
    for (auto& al : mAliasToClassIds) {
      for (auto& classIndex : al.second) {
        alias[al.first] |= triggerMask & (1ul << classIndex);
      }
    }
    // ZDC info
    auto zdc = getZdc(collision.bc(), zdcs);
    bool bbZNA = zdc.timeZNA() > -2. && zdc.timeZNA() < 2.;
    bool bbZNC = zdc.timeZNC() > -2. && zdc.timeZNC() < 2.;
    // VZERO info
    auto vzero = getVZero(collision.bc(), vzeros);
    float timeV0A = vzero.timeA();
    float timeV0C = vzero.timeC();
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
