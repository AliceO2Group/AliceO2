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
/// \brief A tutorial task to retrieve objects from CCDB given a run number.
///        The tutorial shows also how to use timestamps in your analysis.
///        This task requires to access the timestamp table in order to be
///        working. Currently this is done by adding `o2-analysis-timestamp`
///        to the
///        workflow
/// \author Nicolo' Jacazio
/// \since 2020-06-22

#include "Framework/runDataProcessing.h"
#include "Framework/AnalysisTask.h"
#include <CCDB/BasicCCDBManager.h>
#include "CommonDataFormat/InteractionRecord.h"

#include <chrono>

using namespace o2::framework;
using namespace o2::header;
using namespace o2;

struct TimestampUserTask {
  Service<o2::ccdb::BasicCCDBManager> ccdb;
  Configurable<std::string> path{"ccdb-path", "qc/TOF/TOFTaskCompressed/hDiagnostic", "path to the ccdb object"};
  Configurable<std::string> url{"ccdb-url", "http://ccdb-test.cern.ch:8080", "url of the ccdb repository"};
  Configurable<long> nolaterthan{"ccdb-no-later-than", std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count(), "latest acceptable timestamp of creation for the object"};

  void init(o2::framework::InitContext&)
  {
    // Set CCDB url
    ccdb->setURL(url.value);
    // Enabling object caching
    ccdb->setCaching(true);
    // Not later than now objects
    ccdb->setCreatedNotAfter(nolaterthan.value);
  }

  // goup BCs according to Collisions
  void process(aod::BCsWithTimestamps::iterator const& bc, aod::Collisions const& collisions)
  {
    // skip if bc has no Collisions
    LOGF(info, "Size of collisions %i", collisions.size());
    if (collisions.size() == 0) {
      return;
    }

    LOGF(info, "Getting object %s for run number %i from timestamp=%llu", path.value.data(), bc.runNumber(), bc.timestamp());
    // retrieve object for given timestamp
    auto obj = ccdb->getForTimeStamp<TH2F>(path.value, bc.timestamp());
    if (obj) {
      LOGF(info, "Found object!");
      obj->Print("all");
    } else {
      LOGF(warning, "Object not found!");
    }
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const& cfgc)
{
  return WorkflowSpec{adaptAnalysisTask<TimestampUserTask>(cfgc, TaskName{"TimestampUserTask"})};
}
