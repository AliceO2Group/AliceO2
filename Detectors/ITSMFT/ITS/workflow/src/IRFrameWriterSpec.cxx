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

#include "ITSWorkflow/IRFrameWriterSpec.h"
#include "DPLUtils/MakeRootTreeWriterSpec.h"
#include "CommonDataFormat/IRFrame.h"

using namespace o2::framework;

namespace o2
{
namespace its
{

template <typename T>
using BranchDefinition = MakeRootTreeWriterSpec::BranchDefinition<T>;

DataProcessorSpec getIRFrameWriterSpec()
{
  return MakeRootTreeWriterSpec("its-irframe-writer",
                                "o2_its_irframe.root",
                                MakeRootTreeWriterSpec::TreeAttributes{"o2sim", "Tree with reconstructed ITS IR Frames"},
                                BranchDefinition<std::vector<o2::dataformats::IRFrame>>{InputSpec{"irfr", "ITS", "IRFRAMES", 0}, "IRFrames"})();
}

} // namespace its
} // namespace o2
