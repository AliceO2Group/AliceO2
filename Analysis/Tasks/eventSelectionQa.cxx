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

using namespace o2;
using namespace o2::framework;

struct EventSelectionTask {
  aod::FV0A getVZeroA(aod::BC const& bc, aod::FV0As const& vzeros)
  {
    for (auto& vzero : vzeros)
      if (vzero.bc() == bc)
        return vzero;
    aod::FV0A dummy;
    return dummy;
  }

  aod::FV0C getVZeroC(aod::BC const& bc, aod::FV0Cs const& vzeros)
  {
    for (auto& vzero : vzeros)
      if (vzero.bc() == bc)
        return vzero;
    aod::FV0C dummy;
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

  aod::FT0 getFT0(aod::BC const& bc, aod::FT0s const& ft0s)
  {
    for (auto& ft0 : ft0s)
      if (ft0.bc() == bc)
        return ft0;
    aod::FT0 dummy;
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

  void process(soa::Join<aod::Collisions, aod::EvSels>::iterator const& col, aod::BCs const& bcs, aod::Zdcs const& zdcs, aod::FV0As const& fv0as, aod::FV0Cs const& fv0cs, aod::FT0s ft0s, aod::FDDs fdds)
  {
    if (!isMC && !col.alias()[kINT7])
      return;

    auto fv0a = getVZeroA(col.bc(), fv0as);
    auto fv0c = getVZeroC(col.bc(), fv0cs);
    hTimeV0Aall->Fill(fv0a.time());
    hTimeV0Call->Fill(fv0c.time());

    auto zdc = getZdc(col.bc(), zdcs);
    hTimeZNAall->Fill(zdc.timeZNA());
    hTimeZNCall->Fill(zdc.timeZNC());

    auto ft0 = getFT0(col.bc(), ft0s);
    hTimeT0Aall->Fill(ft0.timeA());
    hTimeT0Call->Fill(ft0.timeC());

    auto fdd = getFDD(col.bc(), fdds);
    hTimeFDAall->Fill(fdd.timeA());
    hTimeFDCall->Fill(fdd.timeC());

    if (!col.sel7())
      return;

    hTimeV0Aacc->Fill(fv0a.time());
    hTimeV0Cacc->Fill(fv0c.time());
    hTimeZNAacc->Fill(zdc.timeZNA());
    hTimeZNCacc->Fill(zdc.timeZNC());
    hTimeT0Aacc->Fill(ft0.timeA());
    hTimeT0Cacc->Fill(ft0.timeC());
    hTimeFDAacc->Fill(fdd.timeA());
    hTimeFDCacc->Fill(fdd.timeC());
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const&)
{
  return WorkflowSpec{
    adaptAnalysisTask<EventSelectionTask>("event-selection-qa")};
}
