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

#include "ALICE3GlobalReconstructionWorkflow/RecoWorkflow.h"
#include "ALICE3GlobalReconstructionWorkflow/TrackerSpec.h"
#include "ALICE3GlobalReconstructionWorkflow/TrackWriterSpec.h"
#include "Framework/Logger.h"

namespace o2::trk::global_reco_workflow
{

framework::WorkflowSpec getWorkflow(bool useMC,
                                    const std::string& hitRecoConfig,
                                    const std::string& clusterRecoConfig,
                                    bool disableRootOutput,
                                    o2::gpu::gpudatatypes::DeviceType dtype,
                                    int trackingThreads)
{
  framework::WorkflowSpec specs;

  if (!hitRecoConfig.empty() || !clusterRecoConfig.empty()) {
    LOG_IF(info, !hitRecoConfig.empty()) << "Using hit reco config from file " << hitRecoConfig;
    LOG_IF(info, !clusterRecoConfig.empty()) << "Using cluster reco config from file " << clusterRecoConfig;
    specs.emplace_back(o2::trk::getTrackerSpec(useMC, hitRecoConfig, clusterRecoConfig, dtype, trackingThreads));
    if (!disableRootOutput) {
      specs.emplace_back(o2::trk::getTrackWriterSpec(useMC));
    }
  }

  return specs;
}

} // namespace o2::trk::global_reco_workflow
