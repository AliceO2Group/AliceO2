// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file DPLBroadcaster.cxx
/// \brief Implementation of generic DPL broadcaster, v0.1
///
/// \author Gabriele Gaetano Fronzé, gfronze@cern.ch

#include "DPLUtils/Utils.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/DataProcessingHeader.h"
#include "Headers/DataHeader.h"

namespace o2f = o2::framework;

namespace o2
{
namespace workflows
{

// This is a possible implementation of a DPL compliant and generic broadcaster.
// Every other implementation should fall back to this one, after required translations.
o2f::DataProcessorSpec defineBroadcaster(std::string devName, o2f::InputSpec usrInput, o2f::Outputs usrOutputs,
                                         std::function<size_t(o2f::DataRef)> const func)
{
  return {devName,               // Device name from user
          o2f::Inputs{usrInput}, // User defined input as a vector of one InputSpec
          usrOutputs,            // user defined outputs as a vector of OutputSpecs

          o2f::AlgorithmSpec{[usrOutputs, func](o2f::InitContext&) {
            // Creating shared ptrs to useful parameters
            auto outputsPtr = getOutputList(usrOutputs);
            auto funcPtr = std::make_shared<std::function<size_t(o2f::DataRef)> const>(func);

            // Defining the ProcessCallback as returned object of InitCallback
            return [outputsPtr, funcPtr](o2f::ProcessingContext& ctx) {
              // Getting original input message and getting his size using the provided function
              auto inputMsg = ctx.inputs().getByPos(0);
              // Getting message size using provided std::function
              auto msgSize = (*funcPtr)(inputMsg);
              // Iterating over the OutputSpecs to push the input message to all the output destinations
              for (const auto& itOutputs : (*outputsPtr)) {
                auto& fwdMsg = ctx.outputs().newChunk(itOutputs, msgSize);
                std::memcpy(fwdMsg.data(), inputMsg.payload, msgSize);
              }
            };
          }}};
}

// This is a shortcut for messages with fixed user-defined size
o2f::DataProcessorSpec defineBroadcaster(std::string devName, o2f::InputSpec usrInput, o2f::Outputs usrOutputs,
                                         size_t fixMsgSize)
{
  // This lambda returns a fixed message size
  auto funcSize = [fixMsgSize](o2f::DataRef d) -> size_t { return fixMsgSize; };
  // Callling complete implementation
  return defineBroadcaster(devName, usrInput, usrOutputs, funcSize);
}

// This is an implementation which retrieves the message size using the API
o2f::DataProcessorSpec defineBroadcaster(std::string devName, o2f::InputSpec usrInput, o2f::Outputs usrOutputs)
{
  // This lambda retrieves the message size using the API
  auto funcSize = [](o2f::DataRef d) -> size_t {
    return (o2::header::get<o2::header::DataHeader*>(d.header))->payloadSize;
  };
  // Callling complete implementation
  return defineBroadcaster(devName, usrInput, usrOutputs, funcSize);
}

} // namespace workflows
} // namespace o2
