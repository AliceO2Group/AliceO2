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

/// \file Utils.h
/// \brief A collection of , v0.1
///
/// \author Gabriele Gaetano Fronzé, gfronze@cern.ch

#ifndef UTILS_H
#define UTILS_H

#include "Framework/DataProcessorSpec.h"
#include "MemoryResources/MemoryResources.h"

namespace o2f = o2::framework;

namespace o2
{
namespace workflows
{
//
o2f::Output getOutput(const o2f::OutputSpec outputSpec);
std::shared_ptr<std::vector<o2f::Output>> getOutputList(const o2f::Outputs outputSpecs);

// Broadcaster implementations
o2f::DataProcessorSpec defineBroadcaster(std::string devName, o2f::InputSpec usrInput, o2f::Outputs usrOutputs,
                                         std::function<size_t(o2f::DataRef)> const func);
o2f::DataProcessorSpec defineBroadcaster(std::string devName, o2f::InputSpec usrInput, o2f::Outputs usrOutputs,
                                         size_t fixMsgSize);
o2f::DataProcessorSpec defineBroadcaster(std::string devName, o2f::InputSpec usrInput, o2f::Outputs usrOutputs);

using OutputBuffer = o2::vector<char>;
// Merger implementations
o2f::DataProcessorSpec defineMerger(std::string devName, o2f::Inputs usrInputs, o2f::OutputSpec usrOutput,
                                    std::function<void(OutputBuffer, const o2f::DataRef)> const mergerFunc);
o2f::DataProcessorSpec defineMerger(std::string devName, o2f::Inputs usrInputs, o2f::OutputSpec usrOutput);

// Splitter implementation
o2f::DataProcessorSpec defineRouter(std::string devName, o2f::Inputs usrInput, o2f::Outputs usrOutputs,
                                    std::function<size_t(const o2f::DataRef)> const mappingFunc);

// Gatherer implementation
o2f::DataProcessorSpec defineGatherer(std::string devName, o2f::Inputs usrInputs, o2f::OutputSpec usrOutput);

} // namespace workflows
} // namespace o2

#endif // UTILS_H
