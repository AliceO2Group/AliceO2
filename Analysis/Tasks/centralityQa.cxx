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
#include "Analysis/Centrality.h"
#include "TH1F.h"

using namespace o2;
using namespace o2::framework;
namespace o2
{
namespace aod
{
using CollisionEvSelCent = soa::Join<aod::Collisions, aod::EvSels, aod::Cents>::iterator;
}
} // namespace o2

struct CentralityQaTask {
  OutputObj<TH1F> hCentV0M{TH1F("hCentV0M", "", 21, 0, 105.)};
  void process(aod::CollisionEvSelCent const& col)
  {
    if (!col.alias()[0])
      return;
    if (!col.sel7())
      return;

    LOGF(info, "centV0M=%.0f", col.centV0M());
    // fill centrality histos
    hCentV0M->Fill(col.centV0M());
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const&)
{
  return WorkflowSpec{
    adaptAnalysisTask<CentralityQaTask>("centrality-qa")};
}
