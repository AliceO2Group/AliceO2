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
#include "TFile.h"
#include "TTree.h"
#include "TH1F.h"
#include <map>

using std::map;
using std::string;

using namespace o2;
using namespace o2::framework;

// TODO read nAliases from the alias map?
#define nAliases 2

// TODO move es, mult and cent declarations to AnalysisDataModel.h
namespace o2::aod
{
namespace evsel
{
DECLARE_SOA_INDEX_COLUMN(Collision, collision);
// TODO bool arrays are not supported? Storing in int32 for the moment
DECLARE_SOA_COLUMN(Alias, alias, int32_t[nAliases]);
DECLARE_SOA_COLUMN(BBV0A, bbV0A, bool); // beam-beam time in V0A
DECLARE_SOA_COLUMN(BBV0C, bbV0C, bool); // beam-beam time in V0C
DECLARE_SOA_COLUMN(BGV0A, bgV0A, bool); // beam-gas time in V0A
DECLARE_SOA_COLUMN(BGV0C, bgV0C, bool); // beam-gas time in V0C
DECLARE_SOA_COLUMN(BBZNA, bbZNA, bool); // beam-beam time in ZNA
DECLARE_SOA_COLUMN(BBZNC, bbZNC, bool); // beam-beam time in ZNC
DECLARE_SOA_DYNAMIC_COLUMN(SEL7, sel7, [](bool bbV0A, bool bbV0C, bool bbZNA, bool bbZNC) -> bool { return bbV0A && bbV0C && bbZNA && bbZNC; });

} // namespace evsel
DECLARE_SOA_TABLE(EvSels, "AOD", "ES", evsel::CollisionId,
                  evsel::Alias, evsel::BBV0A, evsel::BBV0C, evsel::BGV0A, evsel::BGV0C, evsel::BBZNA, evsel::BBZNC,
                  evsel::SEL7<evsel::BBV0A, evsel::BBV0C, evsel::BBZNA, evsel::BBZNC>);
using EvSel = EvSels::iterator;

namespace mult
{
DECLARE_SOA_INDEX_COLUMN(Collision, collision);
DECLARE_SOA_COLUMN(MultV0A, multV0A, float);
DECLARE_SOA_COLUMN(MultV0C, multV0C, float);
DECLARE_SOA_DYNAMIC_COLUMN(MultV0M, multV0M, [](float multV0A, float multV0C) -> float { return multV0A + multV0C; });
} // namespace mult
DECLARE_SOA_TABLE(Mults, "AOD", "MULT", mult::CollisionId, mult::MultV0A, mult::MultV0C, mult::MultV0M<mult::MultV0A, mult::MultV0C>);
using Mult = Mults::iterator;

namespace cent
{
DECLARE_SOA_INDEX_COLUMN(Collision, collision);
DECLARE_SOA_COLUMN(CentV0M, centV0M, float);
} // namespace cent
DECLARE_SOA_TABLE(Cents, "AOD", "CENT", cent::CollisionId, cent::CentV0M);
} // namespace o2::aod

aod::Mult getMult(aod::Collision const& collision, aod::Mults const& mults)
{
  for (auto& mult : mults)
    if (mult.collision() == collision)
      return mult;
  aod::Mult dummy;
  return dummy;
}

aod::EvSel getEvSel(aod::Collision const& collision, aod::EvSels const& evsels)
{
  for (auto& evsel : evsels)
    if (evsel.collision() == collision)
      return evsel;
  aod::EvSel dummy;
  return dummy;
}

aod::Trigger getTrigger(aod::Collision const& collision, aod::Triggers const& triggers)
{
  for (auto trigger : triggers)
    if (trigger.globalBC() == collision.globalBC())
      return trigger;
  aod::Trigger dummy;
  return dummy;
}

aod::VZero getVZero(aod::Collision const& collision, aod::VZeros const& vzeros)
{
  // TODO use globalBC to access vzero info
  for (auto& vzero : vzeros)
    if (vzero.collision() == collision)
      return vzero;
  aod::VZero dummy;
  return dummy;
}

aod::Zdc getZdc(aod::Collision const& collision, aod::Zdcs const& zdcs)
{
  // TODO use globalBC to access zdc info
  for (auto& zdc : zdcs)
    if (zdc.collision() == collision)
      return zdc;
  aod::Zdc dummy;
  return dummy;
}

struct EventSelectionTask {
  Produces<aod::EvSels> evsel;
  map<string, int> mClasses;
  map<int, string> mAliases;

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

  void process(aod::Collision const& collision, aod::Triggers const& triggers, aod::Zdcs const& zdcs, aod::VZeros const& vzeros)
  {
    // CTP info
    auto trigger = getTrigger(collision, triggers);
    uint64_t triggerMask = trigger.triggerMask();
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
    auto zdc = getZdc(collision, zdcs);
    // TODO replace it with timing checks when time arrays become available
    uint8_t zdcFired = zdc.fired();
    bool bbZNA = zdcFired & 1 << 0;
    bool bbZNC = zdcFired & 1 << 1;

    // VZERO info
    auto vzero = getVZero(collision, vzeros);
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
    evsel(collision, alias, bbV0A, bbV0C, bgV0A, bgV0C, bbZNA, bbZNC);
  }
};

struct MultiplicityTask {
  Produces<aod::Mults> mult;
  OutputObj<TH1F> hMultV0M{TH1F("hMultV0M", "", 55000, 0, 55000)};

  // TODO use soa::Join for collision + evsel?
  void process(aod::Collision const& collision, aod::VZeros const& vzeros, aod::EvSels const& evsels)
  {
    auto vzero = getVZero(collision, vzeros);
    auto evsel = getEvSel(collision, evsels);

    // VZERO info
    float multV0A = 0;
    float multV0C = 0;
    for (int i = 0; i < 32; i++) {
      // TODO use properly calibrated multiplicity
      multV0A += vzero.adc()[i + 32];
      multV0C += vzero.adc()[i];
    }

    // fill multiplicity columns
    mult(collision, multV0A, multV0C);

    //TODO: bypass alias checks in continuous mode
    if (!evsel.alias()[0] || !evsel.sel7())
      return;

    LOGF(info, "multV0A=%.0f multV0C=%.0f multV0M=%.0f", multV0A, multV0C, multV0A + multV0C);
    // fill calibration histos
    hMultV0M->Fill(multV0A + multV0C);
  }
};

struct CentralityTask {
  Produces<aod::Cents> cent;
  OutputObj<TH1F> hCentV0M{TH1F("hCentV0M", "", 21, 0, 105)};
  TH1F* hCumMultV0M;

  void init(InitContext&)
  {
    // TODO read multiplicity histos from CCDB
    TFile f("centrality.root");
    TH1F* hMultV0M = (TH1F*)f.Get("multiplicity/hMultV0M");
    // TODO produce cumulative histos in the post processing macro
    hCumMultV0M = (TH1F*)hMultV0M->GetCumulative(false);
    hCumMultV0M->Scale(100. / hCumMultV0M->GetMaximum());
  }

  // TODO use soa::Join for collisions,evsels and mults?
  void process(aod::Collision const& collision, aod::EvSels const& evsels, aod::Mults const& mults)
  {
    auto evsel = getEvSel(collision, evsels);
    auto mult = getMult(collision, mults);

    float centV0M = hCumMultV0M->GetBinContent(hCumMultV0M->FindFixBin(mult.multV0M()));
    // fill centrality columns
    cent(collision, centV0M);

    //TODO: bypass alias checks in continuous mode
    if (!evsel.alias()[0] || !evsel.sel7())
      return;

    LOGF(info, "multV0M=%.0f centV0M=%.0f", mult.multV0M(), centV0M);
    // fill centrality histos
    hCentV0M->Fill(centV0M);
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const&)
{
  return WorkflowSpec{
    adaptAnalysisTask<EventSelectionTask>("event-selection"),
    adaptAnalysisTask<MultiplicityTask>("multiplicity"),
    adaptAnalysisTask<CentralityTask>("centrality")};
}
