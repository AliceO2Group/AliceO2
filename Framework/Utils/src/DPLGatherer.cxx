// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file DPLGatherer.cxx
/// \brief Implementation of generic DPL gatherer, v0.1
///
/// \author Gabriele Gaetano Fronzé, gfronze@cern.ch

#include "DPLUtils/Utils.h"
#include "Framework/DataProcessorSpec.h"
#include <vector>

namespace o2f = o2::framework;

namespace o2
{
namespace workflows
{

// This is a possible implementation of a DPL compliant and generic gatherer
o2f::DataProcessorSpec defineGatherer(std::string devName, o2f::Inputs usrInputs, o2f::OutputSpec usrOutput)
{
  return {devName,                 // Device name from user
          usrInputs,               // User defined input as a vector of one InputSpec
          o2f::Outputs{usrOutput}, // user defined outputs as a vector of OutputSpecs

          o2f::AlgorithmSpec{[usrOutput](o2f::InitContext&) {
            // Creating shared ptrs to useful parameters
            auto outputPtr = std::make_shared<o2f::Output>(getOutput(usrOutput));

            // Defining the ProcessCallback as returned object of InitCallback
            return [outputPtr](o2f::ProcessingContext& ctx) {
              // Iterating over the Inputs to forward them on the same Output
              for (const auto& itInputs : ctx.inputs()) {
                // Retrieving message size from API
                auto msgSize = (o2::header::get<o2::header::DataHeader*>(itInputs.header))->payloadSize;
                // Allocating new chunk
                auto& fwdMsg = ctx.outputs().newChunk((*outputPtr), msgSize);
                // Moving the input to the output chunk
                std::memmove(fwdMsg.data(), itInputs.payload, msgSize);
              }
            };
          }}};
}

} // namespace workflows
} // namespace o2
