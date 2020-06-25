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
// A tutorial task to retrieve objects from CCDB given a run number
//
// Author: Nicolo' Jacazio on 2020-06-22

#include "Framework/runDataProcessing.h"
#include "Framework/AnalysisTask.h"
#include "Analysis/RunToTimestamp.h"
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
    ccdb->setCachingEnabled(true);
    // Not later than now objects
    ccdb->setCreatedNotAfter(nolaterthan.value);
  }

  void process(soa::Join<aod::BCs, aod::Timestamps> const& iter)
  {
    for (auto i : iter) {
      auto obj = ccdb->getForTimeStamp<TH2F>(path.value, i.timestamp());
      if (obj) {
        LOGF(info, "Found object!");
        obj->Print("all");
      } else {
        LOGF(warning, "Object not found!");
      }
    }
  }
};

WorkflowSpec defineDataProcessing(ConfigContext const&)
{
  return WorkflowSpec{adaptAnalysisTask<TimestampUserTask>("TimestampUserTask")};
}
