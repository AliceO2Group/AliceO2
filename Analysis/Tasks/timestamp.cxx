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
#include "Framework/ASoA.h"
#include "Framework/ServiceRegistry.h"
#include "Framework/ControlService.h"
#include "Framework/AnalysisTask.h"
#include "Framework/AnalysisDataModel.h"
#include "ReconstructionDataFormats/Track.h"
#include "Framework/ASoAHelpers.h"
#include "DetectorsRaw/HBFUtils.h"

#include <CCDB/CcdbApi.h>

#include <chrono>
#include <thread>

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

  struct asd {
    long GetTimestamp(Int_t run) { return 1; };
  } converter;

  void init(o2::framework::InitContext&)
  {
    o2::ccdb::CcdbApi api;
    LOGF(info, "Initializing TimestampTask");
    Configurable<TString> url{"ccdb-url", "http://ccdb-test.cern.ch:8080", "url of the ccdb repository"};
    Configurable<std::string> path{"ccdb-path", "qc/TOF/TOFTaskCompressed/hDiagnostic", "path to the ccdb object"};
    Configurable<long> timestamp{"ccdb-timestamp", -1, "timestamp of the object"};

    std::map<std::string, std::string> metadata;
    std::map<std::string, std::string>* headers;

    // CCDBobj<TH2F> obj2;
    api.init(url.value.Data());
    if (!api.isHostReachable()) {
      LOGF(warning, "CCDB at URL '%s' is not reachable.", url.value.Data());
      return;
    } else {
      LOGF(info, "Loaded CCDB URL '%s'.", url.value.Data());
    }

    auto obj = api.retrieveFromTFileAny<TH2F>(path.value, metadata, timestamp.value, headers);
    if (obj) {
      LOGF(info, "Found object!");
      // obj->Print("all");
    } else {
      LOGF(warning, "Cannot find object in path '%s'.", path.value.data());
    }
    LOGF(info, "Printing metadata");
    for (auto it = metadata.cbegin(); it != metadata.cend(); ++it) {
      std::cout << it->first << " " << it->second << " " << it->second << "\n";
    }
    // LOGF(info, "Printing headers");
    // for (auto it = headers->cbegin(); it != headers->cend(); ++it) {
    //   std::cout << it->first << " " << it->second << " " << it->second << "\n";
    // }
    Printf("456");
  }
  //   void process(aod::Collision const& collision)
  void process(aod::BC const& bc)
  {
    long timestamp = converter.GetTimestamp(bc.runNumber());
    // timestamp *= 11khz * bc.globalBC() * #bcperorbit;
    ts_table(timestamp);
  }
};

// This is how you can define your processing in a declarative way
WorkflowSpec defineDataProcessing(ConfigContext const&)
{
  return WorkflowSpec{adaptAnalysisTask<TimestampTask>("TimestampTask")};
}
