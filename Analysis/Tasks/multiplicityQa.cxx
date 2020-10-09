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
#include "Analysis/Multiplicity.h"
#include "Analysis/EventSelection.h"
#include "TH1F.h"

using namespace o2;
using namespace o2::framework;

struct MultiplicityQaTask {
  OutputObj<TH1F> hMultV0M{TH1F("hMultV0M", "", 50000, 0., 50000.)};
  OutputObj<TH1F> hMultZNA{TH1F("hMultZNA", "", 600, 0., 240000.)};
  OutputObj<TH1F> hMultZNC{TH1F("hMultZNC", "", 600, 0., 240000.)};

  OutputObj<TProfile> hMultNtrackletsVsV0M{TProfile("hMultNtrackletsVsV0M", "", 50000, 0., 50000.)};

  void process(soa::Join<aod::Collisions, aod::EvSels, aod::Mults>::iterator const& col)
  {
    if (!col.alias()[kINT7])
      return;
    if (!col.sel7())
      return;

    LOGF(debug, "multV0A=%5.0f multV0C=%5.0f multV0M=%5.0f", col.multV0A(), col.multV0C(), col.multV0M());
    // fill calibration histos
    hMultV0M->Fill(col.multV0M());
    hMultZNA->Fill(col.multZNA());
    hMultZNC->Fill(col.multZNC());
    hMultNtrackletsVsV0M->Fill(col.multV0M(), col.multTracklets());
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const&)
{
  return WorkflowSpec{
    adaptAnalysisTask<MultiplicityQaTask>("multiplicity-qa")};
}
