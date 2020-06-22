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
#include "Analysis/RunToTimestamp.h"
#include <CCDB/BasicCCDBManager.h>
#include "CommonDataFormat/InteractionRecord.h"

using namespace o2::framework;
using namespace o2::header;
using namespace o2;

namespace o2::aod
{
namespace ts
{
DECLARE_SOA_INDEX_COLUMN(Collision, collision);
DECLARE_SOA_COLUMN(Timestamp, timestamp, long);
} // namespace ts

DECLARE_SOA_TABLE(TSs, "AOD", "TS", o2::soa::Index<>, ts::Timestamp);

} // namespace o2::aod

struct TimestampTask {
  Produces<aod::TSs> ts_table;
  RunToTimestamp* converter = nullptr;

  void init(o2::framework::InitContext&)
  {
    LOGF(info, "Initializing TimestampTask");
    Service<o2::ccdb::BasicCCDBManager> ccdb;
    Configurable<std::string> path{"ccdb-path", "Test/RunToTimestamp", "path to the ccdb object"};
    Configurable<long> timestamp{"ccdb-timestamp", -1, "timestamp of the object"};
    converter = ccdb->get<RunToTimestamp>(path.value);
    if (converter) {
      LOGF(info, "Run-number to timestamp converter found!");
    } else {
      LOGF(warning, "Cannot find run-number to timestamp converter in path '%s'.", path.value.data());
    }
  }

  void process(aod::BC const& bc)
  {
    long timestamp = converter->getTimestamp(bc.runNumber());
    ts_table(timestamp);
  }
};

// This is how you can define your processing in a declarative way
WorkflowSpec defineDataProcessing(ConfigContext const&)
{
  return WorkflowSpec{adaptAnalysisTask<TimestampTask>("TimestampTask")};
}
