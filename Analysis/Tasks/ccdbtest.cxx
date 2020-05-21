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
#include "Framework/ServiceRegistry.h"
#include "Framework/ControlService.h"
#include "Framework/runDataProcessing.h"
#include "Framework/AnalysisTask.h"
#include "Framework/AnalysisDataModel.h"
#include "ReconstructionDataFormats/Track.h"
#include "PID/PIDResponse.h"
#include "Framework/ASoAHelpers.h"

#include <CCDB/CcdbApi.h>

#include <chrono>
#include <thread>

using namespace o2::framework;
using namespace o2::header;

struct CCDBTask {

  o2::ccdb::CcdbApi api;

  void process(aod::Collision const& collision)
  {
    Configurable<TString> url{"ccdb-url", "http://ccdb-test.cern.ch:8080", "url of the ccdb repository"};
    Configurable<std::string> path{"ccdb-path", "qc/TOF/TOFTaskCompressed/hDiagnostic", "path to the ccdb object"};
    Configurable<long> timestamp{"ccdb-timestamp", -1, "timestamp of the object"};

    std::map<std::string, std::string> metadata;
    std::map<std::string, std::string>* headers;
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
    LOGF(info, "Printing headers");
    for (auto it = headers->cbegin(); it != headers->cend(); ++it) {
      std::cout << it->first << " " << it->second << " " << it->second << "\n";
    }
  }
};

// This is how you can define your processing in a declarative way
WorkflowSpec defineDataProcessing(ConfigContext const&)
{
  return WorkflowSpec{
    adaptAnalysisTask<CCDBTask>("ccdbTestTask")};
  // return WorkflowSpec{
  //   {
  //     "A",
  //     {InputSpec{"somecondition", "TST", "FOO", 0, Lifetime::Condition},
  //      InputSpec{"sometimer", "TST", "BAR", 0, Lifetime::Timer}},
  //     {OutputSpec{"TST", "A1", 0, Lifetime::Timeframe}},
  //     AlgorithmSpec{
  //       adaptStateless([](DataAllocator& outputs, InputRecord& inputs, ControlService& control) {
  //         std::this_thread::sleep_for(std::chrono::milliseconds(1000));
  //         DataRef condition = inputs.get("somecondition");
  //         auto* header = o2::header::get<const DataHeader*>(condition.header);
  //         if (header->payloadSize != 1024) {
  //           LOG(ERROR) << "Wrong size for condition payload (expected " << 1024 << ", found " << header->payloadSize;
  //         }
  //         header->payloadSize;
  //         auto& aData = outputs.make<int>(Output{"TST", "A1", 0}, 1);
  //         control.readyToQuit(QuitRequest::All);
  //       })},
  //     Options{
  //       {"test-option", VariantType::String, "test", {"A test option"}}},
  //   }};
}
