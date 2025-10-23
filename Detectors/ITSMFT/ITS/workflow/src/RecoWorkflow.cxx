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
#include "ITStracking/TrackingConfigParam.h"
#include "ITSMFTWorkflow/DigitReaderSpec.h"
#include "GlobalTrackingWorkflowWriters/IRFrameWriterSpec.h"
#include "GPUWorkflow/GPUWorkflowSpec.h"
#include "Framework/CCDBParamSpec.h"

// Dummy TPC completion policy data

namespace o2::its::reco_workflow
{

framework::WorkflowSpec getWorkflow(bool useMC,
                                    bool useCMtracker,
                                    TrackingMode::Type trmode,
                                    const bool overrideBeamPosition,
                                    bool upstreamDigits,
                                    bool upstreamClusters,
                                    bool disableRootOutput,
                                    bool useGeom,
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
  if ((trmode != TrackingMode::Off) && (TrackerParamConfig::Instance().trackingMode != TrackingMode::Off)) {
    if (useCMtracker) {
      specs.emplace_back(o2::its::getCookedTrackerSpec(useMC, useGeom, useTrig, trmode));
    } else {
      if (useGPUWF) {
        o2::gpu::GPURecoWorkflowSpec::Config cfg{
          .itsTriggerType = useTrig,
          .processMC = useMC,
          .runITSTracking = true,
          .itsOverrBeamEst = overrideBeamPosition,
        };

        Inputs ggInputs;
        auto ggRequest = std::make_shared<o2::base::GRPGeomRequest>(false, true, false, true, true,
                                                                    useGeom ? o2::base::GRPGeomRequest::Aligned : o2::base::GRPGeomRequest::None,
                                                                    ggInputs, true);
        if (!useGeom) {
          ggRequest->addInput({"itsTGeo", "ITS", "GEOMTGEO", 0, Lifetime::Condition, framework::ccdbParamSpec("ITS/Config/Geometry")}, ggInputs);
        }

        static std::vector<InputSpec> policyData;
        static std::shared_ptr<o2::gpu::GPURecoWorkflowSpec> task = std::make_shared<o2::gpu::GPURecoWorkflowSpec>(&policyData, cfg, std::vector<int>(), 0, ggRequest);
        Inputs taskInputs = task->inputs();
        Options taskOptions = task->options();
        std::move(ggInputs.begin(), ggInputs.end(), std::back_inserter(taskInputs));

        specs.emplace_back(DataProcessorSpec{
          .name = "its-gpu-tracker",
          .inputs = taskInputs,
          .outputs = task->outputs(),
          .algorithm = AlgorithmSpec{adoptTask<o2::gpu::GPURecoWorkflowSpec>(task)},
          .options = taskOptions});
      } else {
        specs.emplace_back(o2::its::getTrackerSpec(useMC, useGeom, useTrig, trmode, overrideBeamPosition, dtype));
      }
    }
    if (!disableRootOutput) {
      specs.emplace_back(o2::its::getTrackWriterSpec(useMC));
      specs.emplace_back(o2::globaltracking::getIRFrameWriterSpec("irfr:ITS/IRFRAMES/0", "o2_its_irframe.root", "irframe-writer-its"));
    }
  }
  return specs;
}

} // namespace o2::its::reco_workflow
