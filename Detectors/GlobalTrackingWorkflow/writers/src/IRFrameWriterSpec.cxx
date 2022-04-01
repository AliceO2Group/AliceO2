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

/// @file   IRFrameWriterSpec.cxx

#include <vector>

#include "GlobalTrackingWorkflowWriters/IRFrameWriterSpec.h"
#include "DPLUtils/MakeRootTreeWriterSpec.h"
#include "CommonDataFormat/IRFrame.h"
#include "Framework/DataDescriptorQueryBuilder.h"

using namespace o2::framework;

namespace o2
{
namespace globaltracking
{

template <typename T>
using BranchDefinition = MakeRootTreeWriterSpec::BranchDefinition<T>;

DataProcessorSpec getIRFrameWriterSpec(const std::string& spec, const std::string& defFileName, const std::string& devName)
{
  auto inputs = DataDescriptorQueryBuilder::parse(spec.c_str());
  if (inputs.size() != 1) {
    LOGP(fatal, "irframe-writer expects exactly 1 input spec, {} is received: {}", inputs.size(), spec);
  }
  return MakeRootTreeWriterSpec(devName.c_str(),
                                defFileName.c_str(),
                                MakeRootTreeWriterSpec::TreeAttributes{"o2sim", "Tree with selected IR Frames"},
                                BranchDefinition<std::vector<o2::dataformats::IRFrame>>{inputs.front(), "IRFrames"})();
}

} // namespace globaltracking
} // namespace o2
