// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \author Gabriele Gaetano Fronzé, gfronze@cern.ch

#include "DPLOutputTest.h"
#include <fstream>
#include "DPLUtils/Utils.h"
#include "Framework/DataProcessorSpec.h"
#include "random"
#include "Framework/Logger.h"

namespace o2f = o2::framework;

namespace o2
{
namespace workflows
{

o2f::DataProcessorSpec defineTestGenerator()
{
  return {"Generator",                                                  // Device name
          {},                                                           // No inputs for a generator
          o2f::Outputs{{"TST", "ToSink", 0, o2f::Lifetime::Timeframe}}, // One simple output

          o2f::AlgorithmSpec{[](o2f::InitContext&) {
            int msgCounter = 0;
            auto msgCounter_shptr = std::make_shared<int>(msgCounter);

            LOG(INFO) << ">>>>>>>>>>>>>> Generator initialised\n";

            // Processing context in captured from return on InitCallback
            return [msgCounter_shptr](o2f::ProcessingContext& ctx) {
              int msgIndex = (*msgCounter_shptr)++;
              LOG(INFO) << ">>> MSG:" << msgIndex << "\n";

              LOG(INFO) << ">>> Preparing MSG:" << msgIndex;

              auto& outputMsg = ctx.outputs().newChunk({"TST", "ToSink", 0, o2f::Lifetime::Timeframe},
                                                       (31 + 1) * sizeof(uint32_t) / sizeof(char));

              LOG(INFO) << ">>> Preparing1 MSG:" << msgIndex;

              auto payload = reinterpret_cast<uint32_t*>(outputMsg.data());

              payload[0] = msgIndex;

              LOG(INFO) << ">>> Preparing2 MSG:" << msgIndex;

              for (int k = 0; k < 31; ++k) {
                payload[k + 1] = (uint32_t)k;
                LOG(INFO) << ">>>>\t" << payload[k + 1];
              }

              LOG(INFO) << ">>> Done MSG:" << msgIndex;
            };
          }}};
}

o2f::DataProcessorSpec defineTestSink()
{
  return {"Sink",                                                               // Device name
          o2f::Inputs{{"input", "TST", "ToSink", 0, o2f::Lifetime::Transient}}, // No inputs, for the moment
          {},

          o2f::AlgorithmSpec{[](o2f::InitContext&) {
            LOG(INFO) << ">>>>>>>>>>>>>> Sink initialised\n";

            // Processing context in captured from return on InitCallback
            return [](o2f::ProcessingContext& ctx) {
              auto inputMsg = ctx.inputs().getByPos(0);
              auto payload = reinterpret_cast<const uint32_t*>(inputMsg.payload);

              LOG(INFO) << "Received message containing" << payload[0] << "elements\n";
              for (int j = 0; j < payload[0]; ++j) {
                LOG(INFO) << payload[j];
              }
            };
          }}};
}

o2::framework::WorkflowSpec DPLOutputTest()
{
  auto lspec = o2f::WorkflowSpec();

  // A generator of data
  lspec.emplace_back(defineTestGenerator());
  lspec.emplace_back(defineTestSink());
  return std::move(lspec);
}

} // namespace workflows
} // namespace o2
