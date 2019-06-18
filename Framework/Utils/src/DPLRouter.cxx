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
o2f::DataProcessorSpec defineRouter(std::string devName, o2f::Inputs usrInput, o2f::Outputs usrOutputs,
                                    std::function<size_t(const o2f::DataRef)> const mappingFunc)
{
  return { devName,                 // Device name from user
           o2f::Inputs{ usrInput }, // User defined input as a vector of one InputSpec
           usrOutputs,              // user defined outputs as a vector of OutputSpecs

           o2f::AlgorithmSpec{ [usrOutputs, mappingFunc](o2f::InitContext&) {
             // Creating shared ptrs to useful parameters
             auto outputsPtr = getOutputList(usrOutputs);
             auto mappingFuncPtr = std::make_shared<std::function<size_t(o2f::DataRef)> const>(mappingFunc);

             // Defining the ProcessCallback as returned object of InitCallback
             return [outputsPtr, mappingFuncPtr](o2f::ProcessingContext& ctx) {
               auto inputMsg = ctx.inputs().getByPos(0);
               auto msgSize = (o2::header::get<o2::header::DataHeader*>(inputMsg.header))->payloadSize;
               auto& outputCh = (*outputsPtr)[(*mappingFuncPtr)(inputMsg)];

               auto& fwdMsg = ctx.outputs().newChunk(outputCh, msgSize);
               std::memcpy(fwdMsg.data(), inputMsg.payload, msgSize);
             };
           } } };
}
} // namespace workflows

} // namespace o2
