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
#include "AnalysisDataModel/Centrality.h"
#include "TH1F.h"

using namespace o2;
using namespace o2::framework;

struct CentralityQaTask {
  OutputObj<TH1F> hCentV0M{TH1F("hCentV0M", "", 21, 0, 105.)};
  void process(soa::Join<aod::Collisions, aod::EvSels, aod::Cents>::iterator const& col)
  {
    if (!col.alias()[kINT7]) {
      return;
    }
    if (!col.sel7()) {
      return;
    }

    LOGF(debug, "centV0M=%.0f", col.centV0M());
    // fill centrality histos
    hCentV0M->Fill(col.centV0M());
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  return WorkflowSpec{
    adaptAnalysisTask<CentralityQaTask>(cfgc, TaskName{"centrality-qa"})};
}
