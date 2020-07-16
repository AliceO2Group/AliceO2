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
// Uses RunToTimestamp object from CCDB, fails if not available.
//
// Author: Nicolo' Jacazio on 2020-06-22

#include "Framework/runDataProcessing.h"
#include "Framework/AnalysisTask.h"
#include "Analysis/RunToTimestamp.h"
#include <CCDB/BasicCCDBManager.h>
#include "CommonDataFormat/InteractionRecord.h"
#include "DetectorsRaw/HBFUtils.h"
#include <map>
using namespace o2::framework;
using namespace o2::header;
using namespace o2;

struct TimestampTask {
  Produces<aod::Timestamps> ts_table;
  RunToTimestamp* converter = nullptr;
  std::map<int, int>* mapStartOrbit = nullptr;
  Service<o2::ccdb::BasicCCDBManager> ccdb;
  Configurable<std::string> path{"ccdb-path", "Analysis/Core/RunToTimestamp", "path to the ccdb object"};
  Configurable<long> timestamp{"ccdb-timestamp", -1, "timestamp of the object"};

  void init(o2::framework::InitContext&)
  {
    LOGF(info, "Initializing TimestampTask");
    converter = ccdb->get<RunToTimestamp>(path.value);
    if (converter) {
      LOGF(info, "Run-number to timestamp converter found!");
    } else {
      LOGF(fatal, "Cannot find run-number to timestamp converter in path '%s'.", path.value.data());
    }
    mapStartOrbit = ccdb->get<std::map<int, int>>("Trigger/StartOrbit");
    if (!mapStartOrbit) {
      LOGF(fatal, "Cannot find map of SOR orbits in CCDB in path Trigger/StartOrbit");
    }
  }

  void process(aod::BC const& bc)
  {
    long timestamp = converter->getTimestamp(bc.runNumber());
    uint16_t currentBC = bc.globalBC() % o2::constants::lhc::LHCMaxBunches;
    uint32_t currentOrbit = bc.globalBC() / o2::constants::lhc::LHCMaxBunches;
    uint16_t initialBC = 0; // exact bc number not relevant due to ms precision of timestamps
    uint32_t initialOrbit = mapStartOrbit->at(bc.runNumber());
    InteractionRecord current(currentBC, currentOrbit);
    InteractionRecord initial(initialBC, initialOrbit);
    timestamp += (current - initial).bc2ns() / 1000000;
    ts_table(timestamp);
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const&)
{
  return WorkflowSpec{adaptAnalysisTask<TimestampTask>("TimestampTask")};
}
