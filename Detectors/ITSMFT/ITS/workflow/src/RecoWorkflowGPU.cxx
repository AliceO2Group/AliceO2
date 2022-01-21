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

#include "ITSWorkflow/RecoWorkflowGPU.h"
#include "ITSWorkflow/ClustererSpec.h"
#include "ITSWorkflow/ClusterWriterSpec.h"
#include "ITSWorkflow/IRFrameWriterSpec.h"
#include "ITSWorkflow/TrackerSpecGPU.h"
#include "ITSWorkflow/TrackWriterSpec.h"
#include "ITSMFTWorkflow/EntropyEncoderSpec.h"
#include "ITSMFTWorkflow/DigitReaderSpec.h"

namespace o2
{
namespace its
{
namespace reco_workflow_gpu
{

framework::WorkflowSpec getWorkflow(bool useMC, const std::string& trmode, o2::gpu::GPUDataTypes::DeviceType dtype,
                                    bool upstreamDigits, bool upstreamClusters, bool disableRootOutput, bool eencode)
{
  framework::WorkflowSpec specs;

  if (!(upstreamDigits || upstreamClusters)) {
    specs.emplace_back(o2::itsmft::getITSDigitReaderSpec(useMC, false, "itsdigits.root"));
  }

  if (!upstreamClusters) {
    specs.emplace_back(o2::its::getClustererSpec(useMC));
  }
  if (!disableRootOutput) {
    specs.emplace_back(o2::its::getClusterWriterSpec(useMC));
    specs.emplace_back(o2::its::getTrackWriterSpec(useMC));
    specs.emplace_back(o2::its::getIRFrameWriterSpec());
  }

  specs.emplace_back(o2::its::getTrackerGPUSpec(useMC, trmode, dtype));

  if (eencode) {
    specs.emplace_back(o2::itsmft::getEntropyEncoderSpec("ITS"));
  }
  return specs;
}

} // namespace reco_workflow_gpu
} // namespace its
} // namespace o2
