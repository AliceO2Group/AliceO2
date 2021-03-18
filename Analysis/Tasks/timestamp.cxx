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
  o2::ccdb::CcdbApi ccdb_api;                  /// API to access CCDB
  std::map<int, int>* mapStartOrbit = nullptr; /// Map of the starting orbit for the run
  std::pair<int, long> lastCall;               /// Last run number processed and its timestamp, needed for caching
  std::map<int, long> mapRunToTimestamp;       /// Cache of processed run numbers

  // Configurables
  Configurable<bool> verbose{"verbose", false, "verbose mode"};
  Configurable<std::string> rct_path{"rct-path", "RCT/RunInformation/", "path to the ccdb RCT objects for the SOR timestamps"};
  Configurable<std::string> start_orbit_path{"start-orbit-path", "Trigger/StartOrbit", "path to the ccdb SOR orbit objects"};
  Configurable<std::string> url{"ccdb-url", "http://ccdb-test.cern.ch:8080", "URL of the CCDB database"};
  Configurable<bool> isMC{"isMC", 0, "0 - data, 1 - MC"};

  void init(o2::framework::InitContext&)
  {
    LOGF(info, "Initializing TimestampTask");
    ccdb->setURL(url.value); // Setting URL of CCDB manager from configuration
    LOGF(debug, "Getting SOR orbit map from CCDB url '%s' path '%s'", url.value, start_orbit_path.value);
    mapStartOrbit = ccdb->get<std::map<int, int>>(start_orbit_path.value);
    if (!mapStartOrbit) {
      LOGF(fatal, "Cannot find map of SOR orbits in CCDB in path %s", start_orbit_path.value.data());
    }
    ccdb_api.init(url.value);
    if (!ccdb_api.isHostReachable()) {
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
      const std::string run_path = Form("%s/%i", rct_path.value.data(), bc.runNumber());
      headers = ccdb_api.retrieveHeaders(run_path, metadata, -1);
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
    const uint16_t initialBC = 0; // exact bc number not relevant due to ms precision of timestamps
    if (!mapStartOrbit->count(bc.runNumber())) {
      LOGF(fatal, "Cannot find run %i in mapStartOrbit map", bc.runNumber());
    }
    const uint32_t initialOrbit = mapStartOrbit->at(bc.runNumber());
    const uint16_t currentBC = isMC ? initialBC : bc.globalBC() % o2::constants::lhc::LHCMaxBunches;
    const uint32_t currentOrbit = isMC ? initialOrbit : bc.globalBC() / o2::constants::lhc::LHCMaxBunches;
    const InteractionRecord current(currentBC, currentOrbit);
    const InteractionRecord initial(initialBC, initialOrbit);
    ts += (current - initial).bc2ns() / 1000000;
    ts_table(ts);
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  return WorkflowSpec{adaptAnalysisTask<TimestampTask>(cfgc, TaskName{"TimestampTask"})};
}
