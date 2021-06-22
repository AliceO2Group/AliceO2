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
/// \brief A tutorial task to retrieve objects from CCDB given a run number.
///        This example demonstrates how the CCDB can be used to store an efficiency object which is valid only for a specific time
///        interval (e.g. for a run)
///        The objects are uploaded with https://alimonitor.cern.ch/ccdb/upload.jsp
///        Different timestamps intervals can be given.
///        You need to run this with the o2-analysis-timestamp task
///        NOTE If only one efficiency object for all runs is needed, this code is not optimal.
///        In this case please check the example: efficiencyGlobal.cxx
/// \author
/// \since

#include "Framework/runDataProcessing.h"
#include "Framework/AnalysisTask.h"
#include <CCDB/BasicCCDBManager.h>
#include <chrono>

using namespace o2::framework;
using namespace o2;

struct EfficiencyPerRun {
  Service<ccdb::BasicCCDBManager> ccdb;
  Configurable<std::string> path{"ccdb-path", "Users/j/jgrosseo/tutorial/efficiency/simple", "base path to the ccdb object"};
  Configurable<std::string> url{"ccdb-url", "http://alice-ccdb.cern.ch", "url of the ccdb repository"};
  Configurable<long> nolaterthan{"ccdb-no-later-than", std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count(), "latest acceptable timestamp of creation for the object"};

  OutputObj<TH1F> pt{TH1F("pt", "pt", 20, 0., 10.)};

  void init(InitContext&)
  {
    ccdb->setURL(url.value);
    // Enabling object caching, otherwise each call goes to the CCDB server
    ccdb->setCaching(true);
    ccdb->setLocalObjectValidityChecking();
    // Not later than now, will be replaced by the value of the train creation
    // This avoids that users can replace objects **while** a train is running
    ccdb->setCreatedNotAfter(nolaterthan.value);
  }

  void process(aod::Collision const& collision, aod::BCsWithTimestamps const&, aod::Tracks const& tracks)
  {
    auto bc = collision.bc_as<aod::BCsWithTimestamps>();
    LOGF(info, "Getting object %s for run number %i from timestamp=%llu", path.value.data(), bc.runNumber(), bc.timestamp());
    auto efficiency = ccdb->getForTimeStamp<TH1F>(path.value, bc.timestamp());
    if (!efficiency) {
      LOGF(fatal, "Efficiency object not found!");
    }

    for (auto& track : tracks) {
      pt->Fill(track.pt(), efficiency->GetBinContent(efficiency->FindBin(track.pt())));
      //LOGF(info, "Efficiency %f for pt %f", efficiency->GetBinContent(efficiency->FindBin(track.pt())), track.pt());
    }
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  return WorkflowSpec{
    adaptAnalysisTask<EfficiencyPerRun>(cfgc),
  };
}
