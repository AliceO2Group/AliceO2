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
    ccdb->setCaching(true);
    ccdb->setLocalObjectValidityChecking();
  }

  void process(soa::Join<aod::Collisions, aod::Mults>::iterator const& collision, aod::BCsWithTimestamps const&)
  {
    auto bc = collision.bc_as<aod::BCsWithTimestamps>();
    LOGF(debug, "timestamp=%llu", bc.timestamp());
    TH1F* hCumMultV0M = ccdb->getForTimeStamp<TH1F>("Multiplicity/CumMultV0M", bc.timestamp());
    if (!hCumMultV0M) {
      LOGF(fatal, "V0M centrality calibration is not available in CCDB for run=%d at timestamp=%llu", bc.runNumber(), bc.timestamp());
    }
    float centV0M = hCumMultV0M->GetBinContent(hCumMultV0M->FindFixBin(collision.multV0M()));

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
