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

/// @file   GPUWorkflowITS.cxx
/// @author David Rohr, Matteo Concas

#include "GPUWorkflow/GPUWorkflowSpec.h"
#include "Framework/WorkflowSpec.h" // o2::framework::mergeInputs
#include "Framework/ControlService.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/CCDBParamSpec.h"

#include "ITStracking/TrackingInterface.h"

namespace o2::gpu
{

int32_t GPURecoWorkflowSpec::runITSTracking(o2::framework::ProcessingContext& pc)
{
  mITSTimeFrame->setDevicePropagator(mGPUReco->GetDeviceO2Propagator());
  LOGP(debug, "GPUChainITS is giving me device propagator: {}", (void*)mGPUReco->GetDeviceO2Propagator());
  mITSTrackingInterface->run<true>(pc);
  return 0;
}

void GPURecoWorkflowSpec::initFunctionITS(o2::framework::InitContext& ic)
{
  o2::its::VertexerTraits* vtxTraits = nullptr;
  o2::its::TrackerTraits* trkTraits = nullptr;
  mITSTrackingInterface = std::make_unique<o2::its::ITSTrackingInterface>(mSpecConfig.processMC,
                                                                          mSpecConfig.itsTriggerType,
                                                                          mSpecConfig.itsOverrBeamEst);
  mGPUReco->GetITSTraits(trkTraits, vtxTraits, mITSTimeFrame);
  mITSTrackingInterface->setTraitsFromProvider(vtxTraits, trkTraits, mITSTimeFrame);
  mITSTrackingInterface->initialise();
}

void GPURecoWorkflowSpec::finaliseCCDBITS(o2::framework::ConcreteDataMatcher& matcher, void* obj)
{
  mITSTrackingInterface->finaliseCCDB(matcher, obj);
}

bool GPURecoWorkflowSpec::fetchCalibsCCDBITS(o2::framework::ProcessingContext& pc)
{
  mITSTrackingInterface->updateTimeDependentParams(pc);
  return false;
}
} // namespace o2::gpu
