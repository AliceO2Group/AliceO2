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
// Every other implementation should fall back to this one, after required translations.
o2f::DataProcessorSpec defineMerger(std::string devName, o2f::Inputs usrInputs, o2f::OutputSpec usrOutput,
                                    std::function<void(OutputBuffer, const o2f::DataRef)> const mergerFunc)
{
  return { devName,                   // Device name from user
           usrInputs,                 // User defined input as a vector of one InputSpec
           o2f::Outputs{ usrOutput }, // user defined outputs as a vector of OutputSpecs

           o2f::AlgorithmSpec{ [usrOutput, mergerFunc](o2f::InitContext&) {
             // Creating shared ptrs to useful parameters
             auto outputPtr = std::make_shared<o2f::Output>(getOutput(usrOutput));
             auto mergerFuncPtr = std::make_shared<std::function<void(OutputBuffer, o2f::DataRef)> const>(mergerFunc);

             // Defining the ProcessCallback as returned object of InitCallback
             return [outputPtr, mergerFuncPtr](o2f::ProcessingContext& ctx) {
               OutputBuffer outputBuffer;
               // Iterating over the InputSpecs to aggregate msgs from the connected devices
               for (const auto& itInputs : ctx.inputs()) {
                 (*mergerFuncPtr)(outputBuffer, itInputs);
               }
               // Adopting the buffer as new chunk
               ctx.outputs().adoptChunk((*outputPtr), &outputBuffer[0], outputBuffer.size(), &freefn,
                                        nullptr);
             };
           } } };
}

// This is a possible implementation of a DPL compliant and generic gatherer whit trivial messages concatenation
o2f::DataProcessorSpec defineMerger(std::string devName, o2f::Inputs usrInputs, o2f::OutputSpec usrOutput)
{
  // This lambda retrieves the payload size through the API and back-inserts it on the output buffer
  auto funcMerge = [](OutputBuffer buf, const o2f::DataRef d) {
    auto msgSize = (o2::header::get<o2::header::DataHeader*>(d.header))->payloadSize;
    buf.resize(buf.size() + msgSize);
    std::copy(&(d.payload[0]), &(d.payload[msgSize - 1]), std::back_inserter(buf));
  };
  // Callling complete implementation
  return defineMerger(devName, usrInputs, usrOutput, funcMerge);
}

} // namespace workflows
} // namespace o2
