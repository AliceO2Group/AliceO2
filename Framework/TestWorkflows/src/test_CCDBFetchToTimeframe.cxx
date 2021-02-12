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
#include "Framework/Logger.h"
#include "Framework/ControlService.h"

#include <chrono>
#include <thread>

using namespace o2::framework;
using namespace o2::header;

// Set a start value which might correspond to a real timestamp of an object in CCDB, for example:
// o2-testworkflows-ccdb-fetch-to-timeframe --condition-backend http://ccdb-test.cern.ch:8080 --start-value-enumeration 1575985965925000
WorkflowSpec defineDataProcessing(ConfigContext const&)
{
  return WorkflowSpec{
    {"A",
     {},
     {OutputSpec{"TST", "A1", 0, Lifetime::Timeframe}},
     AlgorithmSpec{
       adaptStateless([](DataAllocator& outputs, InputRecord& inputs, ControlService& control) {
         std::this_thread::sleep_for(std::chrono::milliseconds(1000));
         auto aData = outputs.make<int>(Output{"TST", "A1", 0}, 1);
       })}},
    {
      "B",
      {InputSpec{"somecondition", "TST", "textfile", 0, Lifetime::Condition},
       InputSpec{"somedata", "TST", "A1", 0, Lifetime::Timeframe}},
      {},
      AlgorithmSpec{
        adaptStateless([](DataAllocator& outputs, InputRecord& inputs, ControlService& control) {
          std::this_thread::sleep_for(std::chrono::milliseconds(1000));
          DataRef condition = inputs.get("somecondition");
          auto* header = o2::header::get<const DataHeader*>(condition.header);
          if (header->payloadSize != 1509) {
            LOG(ERROR) << "Wrong size for condition payload (expected " << 1509 << ", found " << header->payloadSize;
          }
          control.readyToQuit(QuitRequest::All);
        })},
    }};
}
