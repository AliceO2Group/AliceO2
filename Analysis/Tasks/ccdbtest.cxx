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
#include "Framework/AnalysisTask.h"
#include "Framework/AnalysisDataModel.h"
#include "ReconstructionDataFormats/Track.h"
#include "Framework/ASoAHelpers.h"

#include <CCDB/CcdbApi.h>
#include <CCDB/BasicCCDBManager.h>

#include <chrono>
#include <thread>

using namespace o2::framework;
using namespace o2::header;
using namespace o2;

struct CCDBTask {
  Service<o2::ccdb::BasicCCDBManager> cdb;
  Configurable<std::string> path{"ccdb-path", "qc/TOF/TOFTaskCompressed/hDiagnostic", "path to the ccdb object"};

  void init(o2::framework::InitContext&)
  {
    Configurable<std::string> url{"ccdb-url", "http://ccdb-test.cern.ch:8080", "url of the ccdb repository"};
    Configurable<long> timestamp{"ccdb-timestamp", -1, "timestamp of the object"};

    cdb->setURL(url);
    cdb->setTimestamp(timestamp);
    cdb->setCachingEnabled(true);
    cdb->setCreatedNotAfter(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count());
  }

  void process(aod::Collision const& collision)
  {
    auto obj = cdb->get<TH2F>(path.value);
    auto obj2 = cdb->getForTimeStamp<TH2F>(path.value, -1);
    if (obj) {
      LOGF(info, "Found object!");
      obj->Print("all");
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
