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

/// @file   RecoWorkflow.cxx

#include "ITSWorkflow/RecoWorkflow.h"
#include "ITSWorkflow/ClustererSpec.h"
#include "ITSWorkflow/ClusterWriterSpec.h"
#include "ITSWorkflow/TrackerSpec.h"
#include "ITSWorkflow/CookedTrackerSpec.h"
#include "ITSWorkflow/TrackWriterSpec.h"
#include "ITSMFTWorkflow/DigitReaderSpec.h"
#include "GlobalTrackingWorkflowWriters/IRFrameWriterSpec.h"
#include "GPUWorkflow/GPUWorkflowSpec.h"

// Dummy TPC completion policy data
using CompletionPolicyData = std::vector<InputSpec>;
static CompletionPolicyData gPolicyData;
static std::shared_ptr<o2::gpu::GPURecoWorkflowSpec> gTask;

namespace o2
{
namespace its
{

namespace reco_workflow
{

framework::WorkflowSpec getWorkflow(bool useMC,
                                    bool useCAtracker,
                                    const std::string& trmode,
                                    const bool overrideBeamPosition,
                                    bool upstreamDigits,
                                    bool upstreamClusters,
                                    bool disableRootOutput,
                                    int useTrig,
                                    bool useGPUWF,
                                    o2::gpu::GPUDataTypes::DeviceType dtype)
{
  framework::WorkflowSpec specs;
  if (!(upstreamDigits || upstreamClusters)) {
    specs.emplace_back(o2::itsmft::getITSDigitReaderSpec(useMC, false, true, "itsdigits.root"));
  }
  if (!upstreamClusters) {
    specs.emplace_back(o2::its::getClustererSpec(useMC));
  }
  if (!disableRootOutput) {
    specs.emplace_back(o2::its::getClusterWriterSpec(useMC));
  }
  if (!trmode.empty()) {
    if (useCAtracker) {
      if (useGPUWF) {
        o2::gpu::GPURecoWorkflowSpec::Config cfg;
        cfg.runITSTracking = true;
        cfg.itsTriggerType = useTrig;
        cfg.itsOverrBeamEst = overrideBeamPosition;
        cfg.itsTrackingMode = trmode == "sync" ? 0 : (trmode == "async" ? 1 : 2);

        Inputs ggInputs;
        auto ggRequest = std::make_shared<o2::base::GRPGeomRequest>(false, true, false, true, true, o2::base::GRPGeomRequest::Aligned, ggInputs, true);

        auto task = std::make_shared<o2::gpu::GPURecoWorkflowSpec>(&gPolicyData, cfg, std::vector<int>(), 0, ggRequest);
        gTask = task;
        Inputs taskInputs = task->inputs();
        Options taskOptions = task->options();
        std::move(ggInputs.begin(), ggInputs.end(), std::back_inserter(taskInputs));

        specs.emplace_back(DataProcessorSpec{
          "its-tracker",
          taskInputs,
          task->outputs(),
          AlgorithmSpec{adoptTask<o2::gpu::GPURecoWorkflowSpec>(task)},
          taskOptions});
      } else {
        specs.emplace_back(o2::its::getTrackerSpec(useMC, useTrig, trmode, overrideBeamPosition, dtype));
      }
    } else {
      specs.emplace_back(o2::its::getCookedTrackerSpec(useMC, useTrig, trmode));
    }
    if (!disableRootOutput) {
      specs.emplace_back(o2::its::getTrackWriterSpec(useMC));
      specs.emplace_back(o2::globaltracking::getIRFrameWriterSpec("irfr:ITS/IRFRAMES/0", "o2_its_irframe.root", "irframe-writer-its"));
    }
  }
  return specs;
}

} // namespace reco_workflow
} // namespace its
} // namespace o2
