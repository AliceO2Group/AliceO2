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

#include <fstream>
#include "DPLBroadcasterMerger.h"
#include "DPLUtils/Utils.h"
#include "Framework/DataProcessorSpec.h"
#include "random"
#include "FairMQLogger.h"
#include <thread>

namespace o2f = o2::framework;

namespace o2
{
namespace workflows
{

o2f::Inputs noInputs{};
o2f::Outputs noOutputs{};

o2f::DataProcessorSpec defineGenerator(o2f::OutputSpec usrOutput)
{
  return { "Generator",               // Device name
           noInputs,                  // No inputs for a generator
           o2f::Outputs{ usrOutput }, // One simple output

           o2f::AlgorithmSpec{ [usrOutput](o2f::InitContext&) {
             int msgCounter = 0;
             auto msgCounter_shptr = std::make_shared<int>(msgCounter);
             auto usrOutput_shptr = std::make_shared<o2f::Output>(getOutput(usrOutput));

             LOG(INFO) << ">>>>>>>>>>>>>> Generator initialised";

             // Processing context in captured from return on InitCallback
             return [usrOutput_shptr, msgCounter_shptr](o2f::ProcessingContext& ctx) {
               int msgIndex = (*msgCounter_shptr)++;
               LOG(INFO) << ">>> MSG:" << msgIndex;
               std::this_thread::sleep_for(std::chrono::milliseconds(1000));

               LOG(INFO) << ">>> Preparing MSG:" << msgIndex;

               auto& outputMsg =
                 ctx.outputs().newChunk(*usrOutput_shptr, (msgIndex + 1) * sizeof(uint32_t) / sizeof(char));

               LOG(INFO) << ">>> Preparing1 MSG:" << msgIndex;

               auto payload = reinterpret_cast<uint32_t*>(outputMsg.data());

               payload[0] = msgIndex;

               LOG(INFO) << ">>> Preparing2 MSG:" << msgIndex;

               for (int k = 0; k < msgIndex; ++k) {
                 payload[k + 1] = (uint32_t)32;
                 LOG(INFO) << ">>>>\t" << payload[k + 1];
               }

               return;
             };
           } } };
}

o2f::DataProcessorSpec definePipeline(std::string devName, o2f::InputSpec usrInput, o2f::OutputSpec usrOutput)
{
  return { devName,                 // Device name
           o2f::Inputs{ usrInput }, // No inputs, for the moment
           o2f::Outputs{ usrOutput }, o2f::AlgorithmSpec{ [usrOutput](o2f::InitContext&) {
             auto output_sharedptr = std::make_shared<o2f::Output>(getOutput(usrOutput));

             // Processing context in captured from return on InitCallback
             return [output_sharedptr](o2f::ProcessingContext& ctx) {
               auto inputMsg = ctx.inputs().getByPos(0);
               auto msgSize = (o2::header::get<o2::header::DataHeader*>(inputMsg.header))->payloadSize;

               auto& fwdMsg = ctx.outputs().newChunk((*output_sharedptr), msgSize);
               std::memcpy(fwdMsg.data(), inputMsg.payload, msgSize);
             };
           } } };
}

o2f::DataProcessorSpec defineSink(o2f::InputSpec usrInput)
{
  return { "Sink",                  // Device name
           o2f::Inputs{ usrInput }, // No inputs, for the moment
           noOutputs,

           o2f::AlgorithmSpec{ [](o2f::InitContext&) {
             // Processing context in captured from return on InitCallback
             return [](o2f::ProcessingContext& ctx) {
               LOG(INFO) << "Received message ";

               auto inputMsg = ctx.inputs().getByPos(0);
               auto payload = reinterpret_cast<const uint32_t*>(inputMsg.payload);

               LOG(INFO) << "Received message containing" << payload[0] << "elements";

               for (int j = 0; j < payload[0]; ++j) {
                 LOG(INFO) << payload[j + 1] << "\t";
               }
               LOG(INFO);
             };
           } } };
}

o2::framework::WorkflowSpec DPLBroadcasterMergerWorkflow()
{
  auto lspec = o2f::WorkflowSpec();

  // A generator of data
  lspec.emplace_back(defineGenerator(o2f::OutputSpec{ "TST", "ToBC", 0, o2f::Lifetime::Timeframe }));

  // A two-way broadcaster
  lspec.emplace_back(defineBroadcaster("Broadcaster",
                                       o2f::InputSpec{ "input", "TST", "ToBC", 0, o2f::Lifetime::Timeframe },
                                       o2f::Outputs{ { "TST", "BCAST0", 0, o2f::Lifetime::Timeframe },
                                                     { "TST", "BCAST1", 0, o2f::Lifetime::Timeframe } }));

  // Two pipeline devices
  lspec.emplace_back(definePipeline("pip0", o2f::InputSpec{ "bc", "TST", "BCAST0", 0, o2f::Lifetime::Timeframe },
                                    o2f::OutputSpec{ "TST", "PIP0", 0, o2f::Lifetime::Timeframe }));
  lspec.emplace_back(definePipeline("pip1", o2f::InputSpec{ "bc", "TST", "BCAST1", 0, o2f::Lifetime::Timeframe },
                                    o2f::OutputSpec{ "TST", "PIP1", 0, o2f::Lifetime::Timeframe }));

  // A gatherer
  lspec.emplace_back(defineMerger("Merger", o2f::Inputs{ { "input1", "TST", "PIP0", 0, o2f::Lifetime::Timeframe },
                                                         { "input2", "TST", "PIP1", 0, o2f::Lifetime::Timeframe } },
                                  o2f::OutputSpec{ "TST", "ToSink", 0, o2f::Lifetime::Timeframe }));

  // A sink which dumps messages
  lspec.emplace_back(defineSink(o2f::InputSpec{ "input", "TST", "ToSink", 0, o2f::Lifetime::Timeframe }));
  return std::move(lspec);
}

} // namespace workflows
} // namespace o2
