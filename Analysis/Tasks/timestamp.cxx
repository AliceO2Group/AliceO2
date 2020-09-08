// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

//
// A task to fill the timestamp table from run number.
// Uses headers from CCDB
//
// Author: Nicolo' Jacazio on 2020-06-22

#include "Framework/runDataProcessing.h"
#include "Framework/AnalysisTask.h"
#include <CCDB/BasicCCDBManager.h>
#include "CommonDataFormat/InteractionRecord.h"
#include "DetectorsRaw/HBFUtils.h"
#include <map>

using namespace o2::framework;
using namespace o2::header;
using namespace o2;

struct TimestampTask {
  Produces<aod::Timestamps> ts_table;          /// Table with SOR timestamps produced by the task
  Service<o2::ccdb::BasicCCDBManager> ccdb;    /// Object manager in CCDB
  o2::ccdb::CcdbApi api;                       /// API to access CCDB
  std::map<int, int>* mapStartOrbit = nullptr; /// Map of the starting orbit for the run
  std::pair<int, long> lastCall;               /// Last run number processed and its timestamp, needed for caching
  std::map<int, long> mapRunToTimestamp;       /// Cache of processed run numbers

  // Configurables
  Configurable<bool> verbose{"verbose", false, "verbose mode"};
  Configurable<std::string> path{"ccdb-path", "RCT/RunInformation/", "path to the ccdb object"};
  Configurable<std::string> url{"ccdb-url", "http://ccdb-test.cern.ch:8080", "URL of the CCDB database"};

  void init(o2::framework::InitContext&)
  {
    LOGF(info, "Initializing TimestampTask");
    mapStartOrbit = ccdb->get<std::map<int, int>>("Trigger/StartOrbit");
    if (!mapStartOrbit) {
      LOGF(fatal, "Cannot find map of SOR orbits in CCDB in path Trigger/StartOrbit");
    }
    api.init(url.value);
    if (!api.isHostReachable()) {
      LOGF(fatal, "CCDB host %s is not reacheable, cannot go forward", url.value.data());
    }
  }

  void process(aod::BC const& bc)
  {
    long ts = 0;
    if (bc.runNumber() == lastCall.first) { // The run number coincides to the last run processed
      LOGF(debug, "Getting timestamp from last call");
      ts = lastCall.second;
    } else if (mapRunToTimestamp.count(bc.runNumber())) { // The run number was already requested before: getting it from cache!
      LOGF(debug, "Getting timestamp from cache");
      ts = mapRunToTimestamp[bc.runNumber()];
    } else { // The run was not requested before: need to acccess CCDB!
      LOGF(debug, "Getting timestamp from CCDB");
      std::map<std::string, std::string> metadata, headers;
      const std::string run_path = Form("%s/%i", path.value.data(), bc.runNumber());
      headers = api.retrieveHeaders(run_path, metadata, -1);
      if (headers.count("SOR") == 0) {
        LOGF(fatal, "Cannot find run-number to timestamp in path '%s'.", run_path.data());
      }
      ts = atol(headers["SOR"].c_str()); // timestamp of the SOR in ms

      // Adding the timestamp to the cache map
      std::pair<std::map<int, long>::iterator, bool> check;
      check = mapRunToTimestamp.insert(std::pair<int, long>(bc.runNumber(), ts));
      if (!check.second) {
        LOGF(fatal, "Run number %i already existed with a timestamp of %llu", bc.runNumber(), check.first->second);
      }
      LOGF(info, "Add new run %i with timestamp %llu to cache", bc.runNumber(), ts);
    }

    // Setting latest run information
    lastCall = std::make_pair(bc.runNumber(), ts);

    if (verbose.value) {
      LOGF(info, "Run-number to timestamp found! %i %llu ms", bc.runNumber(), ts);
    }
    uint16_t currentBC = bc.globalBC() % o2::constants::lhc::LHCMaxBunches;
    uint32_t currentOrbit = bc.globalBC() / o2::constants::lhc::LHCMaxBunches;
    uint16_t initialBC = 0; // exact bc number not relevant due to ms precision of timestamps
    uint32_t initialOrbit = mapStartOrbit->at(bc.runNumber());
    InteractionRecord current(currentBC, currentOrbit);
    InteractionRecord initial(initialBC, initialOrbit);
    ts += (current - initial).bc2ns() / 1000000;
    ts_table(ts);
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const&)
{
  return WorkflowSpec{adaptAnalysisTask<TimestampTask>("TimestampTask")};
}
