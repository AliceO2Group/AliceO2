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
#include "FairMQLogger.h"
#include "Framework/SerializationMethods.h"

using namespace o2::framework;
using DataHeader = o2::header::DataHeader;

/// Example of how to send around strings using DPL.
WorkflowSpec defineDataProcessing(ConfigContext const&)
{
  return WorkflowSpec{
    //
    DataProcessorSpec{
      "boost_serialized_producer", //
      Inputs{},                    //
      {
        OutputSpec{ { "make" }, "TES", "BOOST" }, //
      },
      AlgorithmSpec{ [](ProcessingContext& ctx) {
        auto& out1 = ctx.outputs().make<BoostSerialized<std::array<int, 6>>>({ "TES", "BOOST" });
        // auto& out1 = ctx.outputs().make_boost<std::array<int,6>>({ "TES", "BOOST" });
        // auto& out1 = ctx.outputs().make<BoostSerialized<std::array<int,6>>>({ "TES", "BOOST" });
        // auto& out1 = ctx.outputs().make<std::array<int,6>>({ "TES", "BOOST" });
        out1 = { 1, 2, 3, 4, 5, 6 };
      } } //
    },    //
    DataProcessorSpec{
      "boost_serialized_consumer", //
      {
        InputSpec{ { "make" }, "TES", "BOOST" }, //
      },                                         //
      Outputs{},                                 //
      AlgorithmSpec{
        [](ProcessingContext& ctx) {
          LOG(INFO) << "Buffer ready to receive";

          auto in = ctx.inputs().get<BoostSerialized<std::array<int, 6>>>("make");
          std::array<int, 6> check = { 1, 2, 3, 4, 5, 6 };
          int cnt = 0;
          for (const auto& it : in) {
            if (it != check[cnt]) {
              LOG(ERROR) << "Expecting " << check[cnt] << ", found `" << it << "'";
            } else {
              LOG(INFO) << "Position " << cnt << " OK";
            }
            cnt++;
          }
          ctx.services().get<ControlService>().readyToQuit(true);
        } //
      }   //
    }     //
  };
}
