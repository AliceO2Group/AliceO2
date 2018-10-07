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
#include "DataFormatsMID/Cluster2D.h"

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
        auto& out1 = ctx.outputs().make<BoostSerialized<std::vector<o2::mid::Cluster2D>>>({ "TES", "BOOST" });
        // auto& out1 = ctx.outputs().make_boost<std::array<int,6>>({ "TES", "BOOST" });
        // auto& out1 = ctx.outputs().make<BoostSerialized<std::array<int,6>>>({ "TES", "BOOST" });
        // auto& out1 = ctx.outputs().make<std::array<int,6>>({ "TES", "BOOST" });
        for (size_t i = 0; i < 17; i++) {
          float iFloat = (float)i;
          out1.emplace_back(o2::mid::Cluster2D{ (uint8_t)i, 0.3f * iFloat, 0.5f * iFloat, 0.7f / iFloat, 0.9f / iFloat });
        }
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

          auto in = ctx.inputs().get<BoostSerialized<std::vector<o2::mid::Cluster2D>>>("make");
          std::vector<o2::mid::Cluster2D> check;
          for (size_t i = 0; i < 17; i++) {
            float iFloat = (float)i;
            check.emplace_back(o2::mid::Cluster2D{ (uint8_t)i, 0.3f * iFloat, 0.5f * iFloat, 0.7f / iFloat, 0.9f / iFloat });
          }

          size_t i = 0;
          for (auto const& test : in) {
            assert((test.deId == check[i].deId));       // deId Wrong
            assert((test.xCoor == check[i].xCoor));     // xCoor Wrong
            assert((test.yCoor == check[i].yCoor));     // yCoor Wrong
            assert((test.sigmaX2 == check[i].sigmaX2)); // sigmaX2 Wrong
            assert((test.sigmaY2 == check[i].sigmaY2)); // sigmaY2 Wrong
            i++;
          }
          ctx.services().get<ControlService>().readyToQuit(true);
        } //
      }   //
    }     //
  };
}
