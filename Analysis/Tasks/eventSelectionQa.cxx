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

  OutputObj<TH1F> hTimeV0Aall{TH1F("hTimeV0Aall", "", 200, -50., 50.)};
  OutputObj<TH1F> hTimeV0Call{TH1F("hTimeV0Call", "", 200, -50., 50.)};
  OutputObj<TH1F> hTimeZNAall{TH1F("hTimeZNAall", "", 250, -25., 25.)};
  OutputObj<TH1F> hTimeZNCall{TH1F("hTimeZNCall", "", 250, -25., 25.)};
  OutputObj<TH1F> hTimeV0Aacc{TH1F("hTimeV0Aacc", "", 200, -50., 50.)};
  OutputObj<TH1F> hTimeV0Cacc{TH1F("hTimeV0Cacc", "", 200, -50., 50.)};
  OutputObj<TH1F> hTimeZNAacc{TH1F("hTimeZNAacc", "", 250, -25., 25.)};
  OutputObj<TH1F> hTimeZNCacc{TH1F("hTimeZNCacc", "", 250, -25., 25.)};

  void process(soa::Join<aod::Collisions, aod::EvSels>::iterator const& col, aod::BCs const& bcs, aod::Zdcs const& zdcs, aod::Run2V0s const& vzeros)
  {
    if (!col.alias()[0])
      return;

    auto vzero = getVZero(col.bc(), vzeros);
    hTimeV0Aall->Fill(vzero.timeA());
    hTimeV0Call->Fill(vzero.timeC());

    auto zdc = getZdc(col.bc(), zdcs);
    hTimeZNAall->Fill(zdc.timeZNA());
    hTimeZNCall->Fill(zdc.timeZNC());

    if (!col.sel7())
      return;

    hTimeV0Aacc->Fill(vzero.timeA());
    hTimeV0Cacc->Fill(vzero.timeC());
    hTimeZNAacc->Fill(zdc.timeZNA());
    hTimeZNCacc->Fill(zdc.timeZNC());
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const&)
{
  return WorkflowSpec{
    adaptAnalysisTask<EventSelectionTask>("event-selection-qa")};
}
