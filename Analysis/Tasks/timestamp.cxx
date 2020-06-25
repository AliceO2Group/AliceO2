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

using namespace o2::framework;
using namespace o2::header;
using namespace o2;

struct TimestampTask {
  Produces<aod::Timestamps> ts_table;
  RunToTimestamp* converter = nullptr;
  Service<o2::ccdb::BasicCCDBManager> ccdb;
  Configurable<std::string> path{"ccdb-path", "Test/RunToTimestamp", "path to the ccdb object"};
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
  }

  void process(aod::BC const& bc)
  {
    long timestamp = converter->getTimestamp(bc.runNumber());
    InteractionRecord current(bc.globalBC(), 0);
    InteractionRecord initial = o2::raw::HBFUtils::Instance().getFirstIR();
    timestamp += 1000 * (current - initial).bc2ns();
    ts_table(timestamp);
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const&)
{
  return WorkflowSpec{adaptAnalysisTask<TimestampTask>("TimestampTask")};
}
