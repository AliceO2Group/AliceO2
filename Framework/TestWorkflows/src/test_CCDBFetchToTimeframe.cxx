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
#include "Framework/CCDBParamSpec.h"

#include <chrono>
#include <thread>

using namespace o2::framework;
using namespace o2::header;

// Set a start value which might correspond to a real timestamp of an object in CCDB, for example:
// o2-testworkflows-ccdb-fetch-to-timeframe --condition-backend http://ccdb-test.cern.ch:8080 --start-value-enumeration 1575985965925000
WorkflowSpec defineDataProcessing(ConfigContext const&)
{
  return WorkflowSpec{
    {
      "A",
      {InputSpec{"somecondition", "TST", "textfile", 0, Lifetime::Condition, {ccdbParamSpec("TOF/LHCphase")}},
       InputSpec{"somedata", "TST", "A1", 0, Lifetime::Timer, {startTimeParamSpec(1638548475370)}}},
      {},
      AlgorithmSpec{
        adaptStateless([](DataAllocator& outputs, InputRecord& inputs, ControlService& control) {
          DataRef condition = inputs.get("somecondition");
          auto* header = o2::header::get<const DataHeader*>(condition.header);
          if (header->payloadSize != 2048) {
            LOGP(ERROR, "Wrong size for condition payload (expected {}, found {})", 2048, header->payloadSize);
          }
          control.readyToQuit(QuitRequest::All);
        })},
    }};
}
