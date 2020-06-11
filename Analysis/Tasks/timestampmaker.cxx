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

class TSvsRun : public TNamed
{
 public:
  TSvsRun() = default;
  ~TSvsRun() = default;

  template <typename It>
  void printInsertionStatus(It it, bool success)
  {
    std::cout << "Insertion of " << it->first << (success ? " succeeded\n" : " failed\n");
  }

  Bool_t Has(Int_t run) { return mapping.count(run); }

  void Insert(Int_t run, long timestamp)
  {
    if (Has(run))
      return;
    const auto [it_hinata, success] = mapping.insert({run, timestamp});
    printInsertionStatus(it_hinata, success);
  }
  long GetTimestamp(Int_t run) { return mapping.at(run); }

 private:
  std::map<int, long> mapping;
};

struct TimestampMakerTask {

  o2::ccdb::CcdbApi api;
  //   Configurable<TString> url{"ccdb-url", "http://ccdb-test.cern.ch:8080", "url of the ccdb repository"};
  TString url = "http://ccdb-test.cern.ch:8080";
  //   Configurable<std::string> path{"ccdb-path", "qc/TOF/TOFTaskCompressed/hDiagnostic", "path to the ccdb object"};
  std::string path = "qc/TOF/TOFTaskCompressed/hDiagnostic";
  //   Configurable<long> timestamp{"ccdb-timestamp", -1, "timestamp of the object"};
  long timestamp = -1;

  std::map<std::string, std::string> metadata;
  std::map<std::string, std::string>* headers;
  TSvsRun* converter = nullptr;

  void init(o2::framework::InitContext&)
  {
    api.init(url.Data());
    if (!api.isHostReachable()) {
      LOGF(warning, "CCDB at URL '%s' is not reachable.", url.Data());
      return;
    } else {
      LOGF(info, "Loaded CCDB URL '%s'.", url.Data());
    }

    converter = api.retrieveFromTFileAny<TSvsRun>(path, metadata, timestamp, headers);
    if (!converter) {
      LOGF(info, "Could not get map from CCDB, creating a new one");
      converter = new TSvsRun();
    }
  }

  void process(aod::Collisions const& cols)
  {
    LOGF(info, "Running TimestampMakerTask");
    LOGF(info, "123");
    for (auto& col : cols) {
      auto ir = o2::raw::HBFUtils::Instance().getFirstIR();
      converter->Insert(col.bc().runNumber(), 1234);
    }
    api.storeAsTFileAny(converter, "GRP/RunNumberToTimestamp", metadata, timestamp, timestamp);
  }
};

// This is how you can define your processing in a declarative way
WorkflowSpec defineDataProcessing(ConfigContext const&)
{
  return WorkflowSpec{adaptAnalysisTask<TimestampMakerTask>("TimestampMakerTask")};
}
