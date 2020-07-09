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
#include "Analysis/Centrality.h"
#include <CCDB/BasicCCDBManager.h>
#include "TH1F.h"

using namespace o2;
using namespace o2::framework;

struct CentralityTableTask {
  Produces<aod::Cents> cent;
  Service<o2::ccdb::BasicCCDBManager> ccdb;

  void init(InitContext&)
  {
    ccdb->setURL("http://ccdb-test.cern.ch:8080");
    ccdb->setCachingEnabled(true);
  }

  void process(soa::Join<aod::Collisions, aod::Mults>::iterator const& col, aod::BCs const& bcs)
  {
    TH1F* hCumMultV0M = ccdb->getForTimeStamp<TH1F>("Multiplicity/MultV0M",col.bc().runNumber());
    float centV0M = hCumMultV0M->GetBinContent(hCumMultV0M->FindFixBin(col.multV0M()));
    
    LOGF(debug, "centV0M=%.0f", centV0M);
    // fill centrality columns
    cent(centV0M);
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const&)
{
  return WorkflowSpec{
    adaptAnalysisTask<CentralityTableTask>("centrality-table")};
}
