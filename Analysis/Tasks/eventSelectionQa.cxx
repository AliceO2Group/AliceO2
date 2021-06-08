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
  workflowOptions.push_back(ConfigParamSpec{"selection-run", VariantType::Int, 2, {"selection type: 2 - run 2, 3 - run 3, 0 - run2mc"}});
}

#include "Framework/runDataProcessing.h"
#include "Framework/AnalysisTask.h"
#include "Framework/AnalysisDataModel.h"
#include "AnalysisDataModel/EventSelection.h"
#include "AnalysisCore/TriggerAliases.h"
#include "Framework/HistogramRegistry.h"
#include "TH1F.h"
#include "TH2F.h"

struct EventSelectionQaPerMcCollision {
  HistogramRegistry histos{"HistosPerCollisionMc", {}, OutputObjHandlingPolicy::QAObject};
  void init(InitContext&)
  {
    histos.add("hMcEventCounter", ";;", kTH1F, {{1, 0., 1.}});
  }
  void process(aod::McCollision const& mcCol)
  {
    histos.fill(HIST("hMcEventCounter"), 0.);
  }
};

struct EventSelectionQaPerBc {
  // TODO fill class names in axis labels
  OutputObj<TH1F> hFiredClasses{TH1F("hFiredClasses", "", 100, -0.5, 99.5)};
  OutputObj<TH1F> hFiredAliases{TH1F("hFiredAliases", "", kNaliases, -0.5, kNaliases - 0.5)};
  void init(InitContext&)
  {
    for (int i = 0; i < kNaliases; i++) {
      hFiredAliases->GetXaxis()->SetBinLabel(i + 1, aliasLabels[i]);
    }
  }

  void process(soa::Join<aod::BCs, aod::Run2BCInfos, aod::BcSels>::iterator const& bc)
  {
    // Fill fired trigger classes
    uint64_t triggerMask = bc.triggerMask();
    uint64_t triggerMaskNext50 = bc.triggerMaskNext50();
    for (int i = 0; i < 50; i++) {
      if (triggerMask & 1ull << i) {
        hFiredClasses->Fill(i);
      }
      if (triggerMaskNext50 & 1ull << i) {
        hFiredClasses->Fill(i + 50);
      }
    }

    // Fill fired aliases
    for (int i = 0; i < kNaliases; i++) {
      if (bc.alias()[i]) {
        hFiredAliases->Fill(i);
      }
    }
  }
};

struct EventSelectionQaPerCollision {
  HistogramRegistry histos{"Histos", {}, OutputObjHandlingPolicy::QAObject};

  void init(InitContext&)
  {
    // TODO read low/high flux info from configurable or OADB metadata
    bool isLowFlux = 1;
    for (int i = 0; i < kNaliases; i++) {
      histos.add(Form("%s/hTimeV0Aall", aliasLabels[i]), "All events;V0A;Entries", kTH1F, {{200, -50., 50.}});
      histos.add(Form("%s/hTimeV0Call", aliasLabels[i]), "All events;V0C;Entries", kTH1F, {{200, -50., 50.}});
      histos.add(Form("%s/hTimeZNAall", aliasLabels[i]), "All events;ZNA;Entries", kTH1F, {{250, -25., 25.}});
      histos.add(Form("%s/hTimeZNCall", aliasLabels[i]), "All events;ZNC;Entries", kTH1F, {{250, -25., 25.}});
      histos.add(Form("%s/hTimeT0Aall", aliasLabels[i]), "All events;T0A;Entries", kTH1F, {{200, -10., 10.}});
      histos.add(Form("%s/hTimeT0Call", aliasLabels[i]), "All events;T0C;Entries", kTH1F, {{200, -10., 10.}});
      histos.add(Form("%s/hTimeFDAall", aliasLabels[i]), "All events;FDA;Entries", kTH1F, {{1000, -100., 100.}});
      histos.add(Form("%s/hTimeFDCall", aliasLabels[i]), "All events;FDC;Entries", kTH1F, {{1000, -100., 100.}});
      histos.add(Form("%s/hTimeV0Aacc", aliasLabels[i]), "Accepted events;V0A;Entries", kTH1F, {{200, -50., 50.}});
      histos.add(Form("%s/hTimeV0Cacc", aliasLabels[i]), "Accepted events;V0C;Entries", kTH1F, {{200, -50., 50.}});
      histos.add(Form("%s/hTimeZNAacc", aliasLabels[i]), "Accepted events;ZNA;Entries", kTH1F, {{250, -25., 25.}});
      histos.add(Form("%s/hTimeZNCacc", aliasLabels[i]), "Accepted events;ZNC;Entries", kTH1F, {{250, -25., 25.}});
      histos.add(Form("%s/hTimeT0Aacc", aliasLabels[i]), "Accepted events;T0A;Entries", kTH1F, {{200, -10., 10.}});
      histos.add(Form("%s/hTimeT0Cacc", aliasLabels[i]), "Accepted events;T0C;Entries", kTH1F, {{200, -10., 10.}});
      histos.add(Form("%s/hTimeFDAacc", aliasLabels[i]), "Accepted events;FDA;Entries", kTH1F, {{1000, -100., 100.}});
      histos.add(Form("%s/hTimeFDCacc", aliasLabels[i]), "Accepted events;FDC;Entries", kTH1F, {{1000, -100., 100.}});
      histos.add(Form("%s/hSPDClsVsTklAll", aliasLabels[i]), "All events;n tracklets;n clusters", kTH2F, {{200, 0., isLowFlux ? 200. : 6000.}, {100, 0., isLowFlux ? 100. : 20000.}});
      histos.add(Form("%s/hV0C012vsTklAll", aliasLabels[i]), "All events;n tracklets;V0C012 multiplicity", kTH2F, {{150, 0., 150.}, {150, 0., 600.}});
      histos.add(Form("%s/hV0MOnVsOfAll", aliasLabels[i]), "All events;Offline V0M;Online V0M", kTH2F, {{200, 0., isLowFlux ? 1000. : 50000.}, {400, 0., isLowFlux ? 8000. : 40000.}});
      histos.add(Form("%s/hSPDOnVsOfAll", aliasLabels[i]), "All events;Offline FOR;Online FOR", kTH2F, {{300, 0., isLowFlux ? 300. : 1200.}, {300, 0., isLowFlux ? 300. : 1200.}});
      histos.add(Form("%s/hV0C3vs012All", aliasLabels[i]), "All events;V0C012 multiplicity;V0C3 multiplicity", kTH2F, {{200, 0., 800.}, {300, 0., 300.}});
      histos.add(Form("%s/hSPDClsVsTklAcc", aliasLabels[i]), "Accepted events;n tracklets;n clusters", kTH2F, {{200, 0., isLowFlux ? 200. : 6000.}, {100, 0., isLowFlux ? 100. : 20000.}});
      histos.add(Form("%s/hV0C012vsTklAcc", aliasLabels[i]), "Accepted events;n tracklets;V0C012 multiplicity", kTH2F, {{150, 0., 150.}, {150, 0., 600.}});
      histos.add(Form("%s/hV0MOnVsOfAcc", aliasLabels[i]), "Accepted events;Offline V0M;Online V0M", kTH2F, {{200, 0., isLowFlux ? 1000. : 50000.}, {400, 0., isLowFlux ? 8000. : 40000.}});
      histos.add(Form("%s/hSPDOnVsOfAcc", aliasLabels[i]), "Accepted events;Offline FOR;Online FOR", kTH2F, {{300, 0., isLowFlux ? 300. : 1200.}, {300, 0., isLowFlux ? 300. : 1200.}});
      histos.add(Form("%s/hV0C3vs012Acc", aliasLabels[i]), "Accepted events;V0C012 multiplicity;V0C3 multiplicity", kTH2F, {{200, 0., 800.}, {300, 0., 300.}});
    }
    histos.print();
  }
  Configurable<bool> isMC{"isMC", 0, "0 - data, 1 - MC"};
  Configurable<int> selection{"sel", 7, "trigger: 7 - sel7, 8 - sel8"};

  using BCsWithRun2Infos = soa::Join<aod::BCs, aod::Run2BCInfos>;
  void process(soa::Join<aod::EvSels, aod::Run2MatchedSparse>::iterator const& col,
               BCsWithRun2Infos const& bcs,
               aod::Zdcs const& zdcs,
               aod::FV0As const& fv0as,
               aod::FV0Cs const& fv0cs,
               aod::FT0s ft0s,
               aod::FDDs fdds)
  {
    float timeZNA = col.has_zdc() ? col.zdc().timeZNA() : -999.f;
    float timeZNC = col.has_zdc() ? col.zdc().timeZNC() : -999.f;
    float timeV0A = col.has_fv0a() ? col.fv0a().time() : -999.f;
    float timeV0C = col.has_fv0c() ? col.fv0c().time() : -999.f;
    float timeT0A = col.has_ft0() ? col.ft0().timeA() : -999.f;
    float timeT0C = col.has_ft0() ? col.ft0().timeC() : -999.f;
    float timeFDA = col.has_fdd() ? col.fdd().timeA() : -999.f;
    float timeFDC = col.has_fdd() ? col.fdd().timeC() : -999.f;

    auto bc = col.bc_as<BCsWithRun2Infos>();
    float ofSPD = bc.spdFiredChipsL0() + bc.spdFiredChipsL1();
    float onSPD = bc.spdFiredFastOrL0() + bc.spdFiredFastOrL1();
    float chargeV0M = bc.v0TriggerChargeA() + bc.v0TriggerChargeC();
    float multV0A = col.multRingV0A()[0] + col.multRingV0A()[1] + col.multRingV0A()[2] + col.multRingV0A()[3] + col.multRingV0A()[4];
    float multV0C = col.multRingV0C()[0] + col.multRingV0C()[1] + col.multRingV0C()[2] + col.multRingV0C()[3];
    float multV0M = multV0A + multV0C;
    float multRingV0C3 = col.multRingV0C()[3];
    float multRingV0C012 = multV0C - multRingV0C3;

    histos.fill(HIST("kALL/hTimeV0Aall"), timeV0A);
    histos.fill(HIST("kALL/hTimeV0Call"), timeV0C);
    histos.fill(HIST("kALL/hTimeZNAall"), timeZNA);
    histos.fill(HIST("kALL/hTimeZNCall"), timeZNC);
    histos.fill(HIST("kALL/hTimeT0Aall"), timeT0A);
    histos.fill(HIST("kALL/hTimeT0Call"), timeT0C);
    histos.fill(HIST("kALL/hTimeFDAall"), timeFDA);
    histos.fill(HIST("kALL/hTimeFDCall"), timeFDC);
    histos.fill(HIST("kALL/hSPDClsVsTklAll"), col.spdClusters(), col.nTracklets());
    histos.fill(HIST("kALL/hSPDOnVsOfAll"), ofSPD, onSPD);
    histos.fill(HIST("kALL/hV0MOnVsOfAll"), multV0M, chargeV0M);
    histos.fill(HIST("kALL/hV0C3vs012All"), multRingV0C012, multRingV0C3);
    histos.fill(HIST("kALL/hV0C012vsTklAll"), col.nTracklets(), multRingV0C012);

    // Filling only kINT7 histos for the moment
    // need dynamic histo names in the fill function
    if (col.alias()[kINT7]) {
      histos.fill(HIST("kINT7/hTimeV0Aall"), timeV0A);
      histos.fill(HIST("kINT7/hTimeV0Call"), timeV0C);
      histos.fill(HIST("kINT7/hTimeZNAall"), timeZNA);
      histos.fill(HIST("kINT7/hTimeZNCall"), timeZNC);
      histos.fill(HIST("kINT7/hTimeT0Aall"), timeT0A);
      histos.fill(HIST("kINT7/hTimeT0Call"), timeT0C);
      histos.fill(HIST("kINT7/hTimeFDAall"), timeFDA);
      histos.fill(HIST("kINT7/hTimeFDCall"), timeFDC);
      histos.fill(HIST("kINT7/hSPDClsVsTklAll"), col.spdClusters(), col.nTracklets());
      histos.fill(HIST("kINT7/hSPDOnVsOfAll"), ofSPD, onSPD);
      histos.fill(HIST("kINT7/hV0MOnVsOfAll"), multV0M, chargeV0M);
      histos.fill(HIST("kINT7/hV0C3vs012All"), multRingV0C012, multRingV0C3);
      histos.fill(HIST("kINT7/hV0C012vsTklAll"), col.nTracklets(), multRingV0C012);

      LOGF(info, "selection[kIsBBV0A]=%i", col.selection()[aod::kIsBBV0A]);
      LOGF(info, "selection[kIsBBV0C]=%i", col.selection()[aod::kIsBBV0C]);
      LOGF(info, "selection[kIsBBZNA]=%i", col.selection()[aod::kIsBBZNA]);
      LOGF(info, "selection[kIsBBZNC]=%i", col.selection()[aod::kIsBBZNC]);
      LOGF(info, "selection[kNoTPCHVdip]=%i", col.selection()[aod::kNoTPCHVdip]);
      LOGF(info, "selection[kIsGoodTimeRange]=%i", col.selection()[aod::kIsGoodTimeRange]);
      LOGF(info, "selection[kNoIncompleteDAQ]=%i", col.selection()[aod::kNoIncompleteDAQ]);
      LOGF(info, "selection[kNoV0C012vsTklBG]=%i", col.selection()[aod::kNoV0C012vsTklBG]);
      LOGF(info, "selection[kNoV0Casymmetry]=%i", col.selection()[aod::kNoV0Casymmetry]);
      LOGF(info, "selection[kNoSPDClsVsTklBG]=%i", col.selection()[aod::kNoSPDClsVsTklBG]);
      LOGF(info, "selection[kNoV0MOnVsOfPileup]=%i", col.selection()[aod::kNoV0MOnVsOfPileup]);
      LOGF(info, "selection[kNoSPDOnVsOfPileup]=%i", col.selection()[aod::kNoSPDOnVsOfPileup]);
      LOGF(info, "selection[kNoPileupFromSPD]=%i", col.selection()[aod::kNoPileupFromSPD]);
    }

    if (selection == 7 && !col.sel7()) {
      return;
    }

    if (selection == 8 && !col.sel8()) {
      return;
    }

    if (selection != 7 && selection != 8) {
      LOGF(fatal, "Unknown selection type! Use `--sel 7` or `--sel 8`");
    }

    histos.fill(HIST("kALL/hTimeV0Aacc"), timeV0A);
    histos.fill(HIST("kALL/hTimeV0Cacc"), timeV0C);
    histos.fill(HIST("kALL/hTimeZNAacc"), timeZNA);
    histos.fill(HIST("kALL/hTimeZNCacc"), timeZNC);
    histos.fill(HIST("kALL/hTimeT0Aacc"), timeT0A);
    histos.fill(HIST("kALL/hTimeT0Cacc"), timeT0C);
    histos.fill(HIST("kALL/hTimeFDAacc"), timeFDA);
    histos.fill(HIST("kALL/hTimeFDCacc"), timeFDC);
    histos.fill(HIST("kALL/hSPDClsVsTklAcc"), col.spdClusters(), col.nTracklets());
    histos.fill(HIST("kALL/hSPDOnVsOfAcc"), ofSPD, onSPD);
    histos.fill(HIST("kALL/hV0MOnVsOfAcc"), multV0M, chargeV0M);
    histos.fill(HIST("kALL/hV0C3vs012Acc"), multRingV0C012, multRingV0C3);
    histos.fill(HIST("kALL/hV0C012vsTklAcc"), col.nTracklets(), multRingV0C012);

    if (col.alias()[kINT7]) {
      histos.fill(HIST("kINT7/hTimeV0Aacc"), timeV0A);
      histos.fill(HIST("kINT7/hTimeV0Cacc"), timeV0C);
      histos.fill(HIST("kINT7/hTimeZNAacc"), timeZNA);
      histos.fill(HIST("kINT7/hTimeZNCacc"), timeZNC);
      histos.fill(HIST("kINT7/hTimeT0Aacc"), timeT0A);
      histos.fill(HIST("kINT7/hTimeT0Cacc"), timeT0C);
      histos.fill(HIST("kINT7/hTimeFDAacc"), timeFDA);
      histos.fill(HIST("kINT7/hTimeFDCacc"), timeFDC);
      histos.fill(HIST("kINT7/hSPDClsVsTklAcc"), col.spdClusters(), col.nTracklets());
      histos.fill(HIST("kINT7/hSPDOnVsOfAcc"), ofSPD, onSPD);
      histos.fill(HIST("kINT7/hV0MOnVsOfAcc"), multV0M, chargeV0M);
      histos.fill(HIST("kINT7/hV0C3vs012Acc"), multRingV0C012, multRingV0C3);
      histos.fill(HIST("kINT7/hV0C012vsTklAcc"), col.nTracklets(), multRingV0C012);
    }
  }
};

struct EventSelectionQaPerBcRun3 {
  HistogramRegistry histos{"HistosPerBC", {}, OutputObjHandlingPolicy::QAObject};
  void init(InitContext&)
  {
    histos.add("hTimeT0Aall", "All events;T0A time;Entries", kTH1F, {{200, -2., 2.}});
    histos.add("hTimeT0Call", "All events;T0C time;Entries", kTH1F, {{200, -2., 2.}});
    histos.add("hGlobalBcFT0", ";;", kTH1F, {{10000, 0., 10000.}});
  }

  void process(aod::FT0 ft0, aod::BCs const& bcs)
  {
    float timeA = ft0.timeA();
    float timeC = ft0.timeC();
    histos.fill(HIST("hTimeT0Aall"), timeA);
    histos.fill(HIST("hTimeT0Call"), timeC);
    histos.fill(HIST("hGlobalBcFT0"), ft0.bc().globalBC());
  }
};

struct EventSelectionQaPerCollisionRun3 {
  HistogramRegistry histos{"Histos", {}, OutputObjHandlingPolicy::QAObject};
  void init(InitContext&)
  {
    histos.add("hNcontrib", ";n contributors;", kTH1F, {{100, 0, 100.}});
    histos.add("hNcontribFT0", ";n contributors;", kTH1F, {{100, 0, 100.}});
    histos.add("hGlobalBcCol", ";;", kTH1F, {{10000, 0., 10000.}});
    histos.add("hGlobalBcColFT0", ";;", kTH1F, {{10000, 0., 10000.}});
  }
  void process(soa::Join<aod::Collisions, aod::EvSels>::iterator const& col, aod::BCs const& bcs)
  {
    int nContributors = col.numContrib();
    histos.fill(HIST("hNcontrib"), nContributors);
    uint64_t globalBc = col.bc().globalBC();
    histos.fill(HIST("hGlobalBcCol"), globalBc);
    if (col.foundFT0() >= 0) {
      histos.fill(HIST("hNcontribFT0"), nContributors);
      histos.fill(HIST("hGlobalBcColFT0"), globalBc);
    }
  }
};

struct EventSelectionQaPerMcCollisionRun3 {
  HistogramRegistry histos{"HistosPerCollisionMc", {}, OutputObjHandlingPolicy::QAObject};
  void init(InitContext&)
  {
    histos.add("hMcEventCounter", ";;", kTH1F, {{1, 0., 1.}});
    histos.add("hGlobalBcMcCol", ";;", kTH1F, {{10000, 0., 10000.}});
  }
  void process(aod::McCollision const& mcCol, aod::BCs const& bcs)
  {
    histos.fill(HIST("hMcEventCounter"), 0.);
    uint64_t globalBc = mcCol.bc().globalBC();
    histos.fill(HIST("hGlobalBcMcCol"), globalBc);
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  if (cfgc.options().get<int>("selection-run") == 2) {
    return WorkflowSpec{
      adaptAnalysisTask<EventSelectionQaPerBc>(cfgc),
      adaptAnalysisTask<EventSelectionQaPerCollision>(cfgc)};
  } else if (cfgc.options().get<int>("selection-run") == 0) {
    return WorkflowSpec{
      adaptAnalysisTask<EventSelectionQaPerBc>(cfgc),
      adaptAnalysisTask<EventSelectionQaPerCollision>(cfgc),
      adaptAnalysisTask<EventSelectionQaPerMcCollision>(cfgc)};
  } else {
    return WorkflowSpec{
      adaptAnalysisTask<EventSelectionQaPerBcRun3>(cfgc),
      adaptAnalysisTask<EventSelectionQaPerCollisionRun3>(cfgc),
      adaptAnalysisTask<EventSelectionQaPerMcCollisionRun3>(cfgc)};
  }
}
