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

#include "ITS3Workflow/RecoWorkflow.h"
#include "ITS3Workflow/ClustererSpec.h"
#include "ITS3Workflow/TrackerSpec.h"
#include "ITSMFTWorkflow/ClusterWriterSpec.h"
#include "ITSWorkflow/TrackWriterSpec.h"
#include "ITS3Workflow/DigitReaderSpec.h"
#include "GPUWorkflow/GPUWorkflowSpec.h"
#include "Framework/CCDBParamSpec.h"

// Dummy pointers
using CompletionPolicyData = std::vector<InputSpec>;
static CompletionPolicyData gPolicyData;
static std::shared_ptr<o2::gpu::GPURecoWorkflowSpec> gTask;

namespace o2::its3::reco_workflow
{

framework::WorkflowSpec getWorkflow(bool useMC, its::TrackingMode::Type trmode, o2::gpu::gpudatatypes::DeviceType dtype, bool useGPUWorkflow,
                                    bool upstreamDigits, bool upstreamClusters, bool disableRootOutput, bool useGeom, int useTrig, bool overrideBeamPosition)
{
  framework::WorkflowSpec specs;

  if (!(upstreamDigits || upstreamClusters)) {
    specs.emplace_back(o2::its3::getITS3DigitReaderSpec(useMC, false, "it3digits.root"));
  }

  if (!upstreamClusters) {
    specs.emplace_back(o2::its3::getClustererSpec(useMC));
  }

  if (!disableRootOutput) {
    specs.emplace_back(o2::itsmft::getITSClusterWriterSpec(useMC, false));
  }

  if (trmode != its::TrackingMode::Off) {
    if (useGPUWorkflow) {
      o2::gpu::GPURecoWorkflowSpec::Config cfg;
      cfg.runITSTracking = true;
      cfg.isITS3 = true;
      cfg.itsTriggerType = useTrig;
      cfg.itsOverrBeamEst = overrideBeamPosition;
      cfg.processMC = useMC;
      Inputs ggInputs;
      auto ggRequest = std::make_shared<o2::base::GRPGeomRequest>(false, true, false, true, true,
                                                                  useGeom ? o2::base::GRPGeomRequest::Aligned : o2::base::GRPGeomRequest::None,
                                                                  ggInputs, true);
      if (!useGeom) {
        ggRequest->addInput({"itsTGeo", "ITS", "GEOMTGEO", 0, Lifetime::Condition, framework::ccdbParamSpec("ITS/Config/Geometry")}, ggInputs);
      }

      auto task = std::make_shared<o2::gpu::GPURecoWorkflowSpec>(&gPolicyData, cfg, std::vector<int>(), 0, ggRequest);
      gTask = task;
      Inputs taskInputs = task->inputs();
      Options taskOptions = task->options();
      std::move(ggInputs.begin(), ggInputs.end(), std::back_inserter(taskInputs));

      specs.emplace_back(DataProcessorSpec{
        "its3-gpu-tracker",
        taskInputs,
        task->outputs(),
        AlgorithmSpec{adoptTask<o2::gpu::GPURecoWorkflowSpec>(task)},
        taskOptions});
    } else {
      specs.emplace_back(o2::its3::getTrackerSpec(useMC, useGeom, useTrig, trmode, overrideBeamPosition, dtype));
    }
    if (!disableRootOutput) {
      specs.emplace_back(o2::its::getTrackWriterSpec(useMC));
    }
  }

  return specs;
}

} // namespace o2::its3::reco_workflow
