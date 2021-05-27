// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
///
/// \brief This example demonstrates how the CCDB can be used to store an efficiency object which is valid for a full train run
///        The objects are uploaded with
///        The objects are uploaded with https://alimonitor.cern.ch/ccdb/upload.jsp
///        A sufficiently large time stamp interval should be given to span all runs under consideration
///        NOTE If an efficiency object per run is needed, please check the example efficiencyPerRun.cxx
/// \author
/// \since

#include "Framework/runDataProcessing.h"
#include "Framework/AnalysisTask.h"
#include <CCDB/BasicCCDBManager.h>
#include <chrono>

using namespace o2::framework;
using namespace o2;

struct EfficiencyGlobal {
  Service<ccdb::BasicCCDBManager> ccdb;
  Configurable<std::string> path{"ccdb-path", "Users/j/jgrosseo/tutorial/efficiency/simple", "base path to the ccdb object"};
  Configurable<std::string> url{"ccdb-url", "http://alice-ccdb.cern.ch", "url of the ccdb repository"};
  Configurable<long> nolaterthan{"ccdb-no-later-than", std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count(), "latest acceptable timestamp of creation for the object"};

  OutputObj<TH1F> pt{TH1F("pt", "pt", 20, 0., 10.)};

  // the efficiency has been previously stored in the CCDB as TH1F histogram
  TH1F* efficiency = nullptr;

  void init(InitContext&)
  {
    ccdb->setURL(url.value);
    // Enabling object caching, otherwise each call goes to the CCDB server
    ccdb->setCaching(true);
    ccdb->setLocalObjectValidityChecking();
    // Not later than now, will be replaced by the value of the train creation
    // This avoids that users can replace objects **while** a train is running
    ccdb->setCreatedNotAfter(nolaterthan.value);
    LOGF(info, "Getting object %s", path.value.data());
    efficiency = ccdb->getForTimeStamp<TH1F>(path.value, nolaterthan.value);
    if (!efficiency) {
      LOGF(fatal, "Efficiency object not found!");
    }
  }

  void process(aod::Collision const& collision, aod::Tracks const& tracks)
  {
    for (auto& track : tracks) {
      pt->Fill(track.pt(), efficiency->GetBinContent(efficiency->FindBin(track.pt())));
      //LOGF(info, "Efficiency %f for pt %f", efficiency->GetBinContent(efficiency->FindBin(track.pt())), track.pt());
    }
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  return WorkflowSpec{
    adaptAnalysisTask<EfficiencyGlobal>(cfgc),
  };
}
