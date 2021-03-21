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
#include "AnalysisDataModel/EventSelection.h"

using namespace o2;
using namespace o2::framework;

struct EventSelectionQaPerBc {
  OutputObj<TH1F> hFiredClasses{TH1F("hFiredClasses", "", 100, -0.5, 99.5)};
  void process(soa::Join<aod::BCs, aod::Run2BCInfos>::iterator const& bc)
  {
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
  }
};

struct EventSelectionQaPerCollision {
  OutputObj<TH1F> hTimeV0Aall{TH1F("hTimeV0Aall", "", 200, -50., 50.)};
  OutputObj<TH1F> hTimeV0Call{TH1F("hTimeV0Call", "", 200, -50., 50.)};
  OutputObj<TH1F> hTimeZNAall{TH1F("hTimeZNAall", "", 250, -25., 25.)};
  OutputObj<TH1F> hTimeZNCall{TH1F("hTimeZNCall", "", 250, -25., 25.)};
  OutputObj<TH1F> hTimeT0Aall{TH1F("hTimeT0Aall", "", 200, -10., 10.)};
  OutputObj<TH1F> hTimeT0Call{TH1F("hTimeT0Call", "", 200, -10., 10.)};
  OutputObj<TH1F> hTimeFDAall{TH1F("hTimeFDAall", "", 1000, -100., 100.)};
  OutputObj<TH1F> hTimeFDCall{TH1F("hTimeFDCall", "", 1000, -100., 100.)};
  OutputObj<TH1F> hTimeV0Aacc{TH1F("hTimeV0Aacc", "", 200, -50., 50.)};
  OutputObj<TH1F> hTimeV0Cacc{TH1F("hTimeV0Cacc", "", 200, -50., 50.)};
  OutputObj<TH1F> hTimeZNAacc{TH1F("hTimeZNAacc", "", 250, -25., 25.)};
  OutputObj<TH1F> hTimeZNCacc{TH1F("hTimeZNCacc", "", 250, -25., 25.)};
  OutputObj<TH1F> hTimeT0Aacc{TH1F("hTimeT0Aacc", "", 200, -10., 10.)};
  OutputObj<TH1F> hTimeT0Cacc{TH1F("hTimeT0Cacc", "", 200, -10., 10.)};
  OutputObj<TH1F> hTimeFDAacc{TH1F("hTimeFDAacc", "", 1000, -100., 100.)};
  OutputObj<TH1F> hTimeFDCacc{TH1F("hTimeFDCacc", "", 1000, -100., 100.)};

  Configurable<bool> isMC{"isMC", 0, "0 - data, 1 - MC"};
  Configurable<int> selection{"sel", 7, "trigger: 7 - sel7, 8 - sel8"};

  void process(soa::Join<aod::Collisions, aod::EvSels, aod::Run2MatchedSparse>::iterator const& col,
               aod::Zdcs const& zdcs,
               aod::FV0As const& fv0as,
               aod::FV0Cs const& fv0cs,
               aod::FT0s ft0s,
               aod::FDDs fdds)
  {
    if (!isMC && !col.alias()[kINT7]) {
      return;
    }

    float timeZNA = col.has_zdc() ? col.zdc().timeZNA() : -999.f;
    float timeZNC = col.has_zdc() ? col.zdc().timeZNC() : -999.f;
    float timeV0A = col.has_fv0a() ? col.fv0a().time() : -999.f;
    float timeV0C = col.has_fv0c() ? col.fv0c().time() : -999.f;
    float timeT0A = col.has_ft0() ? col.ft0().timeA() : -999.f;
    float timeT0C = col.has_ft0() ? col.ft0().timeC() : -999.f;
    float timeFDA = col.has_fdd() ? col.fdd().timeA() : -999.f;
    float timeFDC = col.has_fdd() ? col.fdd().timeC() : -999.f;

    hTimeV0Aall->Fill(timeV0A);
    hTimeV0Call->Fill(timeV0C);
    hTimeZNAall->Fill(timeZNA);
    hTimeZNCall->Fill(timeZNC);
    hTimeT0Aall->Fill(timeT0A);
    hTimeT0Call->Fill(timeT0C);
    hTimeFDAall->Fill(timeFDA);
    hTimeFDCall->Fill(timeFDC);

    if (selection == 7 && !col.sel7()) {
      return;
    }

    if (selection == 8 && !col.sel8()) {
      return;
    }

    if (selection != 7 && selection != 8) {
      LOGF(fatal, "Unknown selection type! Use `--sel 7` or `--sel 8`");
    }

    hTimeV0Aacc->Fill(timeV0A);
    hTimeV0Cacc->Fill(timeV0C);
    hTimeZNAacc->Fill(timeZNA);
    hTimeZNCacc->Fill(timeZNC);
    hTimeT0Aacc->Fill(timeT0A);
    hTimeT0Cacc->Fill(timeT0C);
    hTimeFDAacc->Fill(timeFDA);
    hTimeFDCacc->Fill(timeFDC);
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  return WorkflowSpec{
    adaptAnalysisTask<EventSelectionQaPerBc>(cfgc),
    adaptAnalysisTask<EventSelectionQaPerCollision>(cfgc)};
}
