// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#include "Framework/DataRefUtils.h"
#include "Framework/AlgorithmSpec.h"
#include "Framework/ServiceRegistry.h"
#include "Framework/runDataProcessing.h"
#include <Monitoring/Monitoring.h>
#include "Framework/ControlService.h"
#include "Framework/CallbackService.h"
#include "Framework/Logger.h"

using namespace o2::framework;
using DataHeader = o2::header::DataHeader;

/// Example of how to send around strings using DPL.
WorkflowSpec defineDataProcessing(ConfigContext const&)
{
  return WorkflowSpec{
    //
    DataProcessorSpec{
      "string_producer", //
      Inputs{},          //
      {
        OutputSpec{{"make"}, "TES", "STRING"}, //
      },
      AlgorithmSpec{[](ProcessingContext& ctx) {
        auto& out1 = ctx.outputs().make<std::string>(Output{"TES", "STRING"}, "default");
        assert(out1 == "default");
        out1 = "Hello";
      }} //
    },   //
    DataProcessorSpec{
      "string_consumer", //
      {
        InputSpec{"make", "TES", "STRING"}, //
      },                                    //
      Outputs{},                            //
      AlgorithmSpec{
        [](ProcessingContext& ctx) {
          auto s = ctx.inputs().get<std::string>("make");

          if (s != "Hello") {
            LOG(ERROR) << "Expecting `Hello', found `" << s << "'";
          } else {
            LOG(INFO) << "Everything OK";
          }
          ctx.services().get<ControlService>().readyToQuit(true);
        } //
      }   //
    }     //
  };
}
