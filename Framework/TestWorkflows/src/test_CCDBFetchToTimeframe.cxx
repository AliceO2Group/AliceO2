// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#include "Framework/runDataProcessing.h"
#include "Framework/Logger.h"
#include "Framework/ControlService.h"
#include "Framework/CCDBParamSpec.h"
#include "Framework/DataRefUtils.h"

#include <chrono>
#include <thread>

using namespace o2::framework;

// Set a start value which might correspond to a real timestamp of an object in CCDB, for example:
// o2-testworkflows-ccdb-fetch-to-timeframe --condition-backend https://alice-ccdb.cern.ch --start-value-enumeration 1575985965925000
WorkflowSpec defineDataProcessing(ConfigContext const&)
{
  return WorkflowSpec{
    {
      "A",
      {InputSpec{"somecondition", "TST", "textfile", 0, Lifetime::Condition, ccdbParamSpec("TOF/LHCphase")},
       InputSpec{"somedata", "TST", "A1", 0, Lifetime::Timer, {startTimeParamSpec(1638548475370)}}},
      {},
      AlgorithmSpec{
        adaptStateless([](DataAllocator& outputs, InputRecord& inputs, ControlService& control) {
          DataRef condition = inputs.get("somecondition");
          auto payloadSize = DataRefUtils::getPayloadSize(condition);
          if (payloadSize != 2048) {
            LOGP(error, "Wrong size for condition payload (expected {}, found {})", 2048, payloadSize);
          }
          control.readyToQuit(QuitRequest::All);
        })},
    }};
}
