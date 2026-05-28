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

#ifndef O2_ITS3_RECOWORKFLOW_H
#define O2_ITS3_RECOWORKFLOW_H

/// @file   RecoWorkflow.h

#include "Framework/WorkflowSpec.h"
#include "ITStracking/Configuration.h"
#include "GPUDataTypesConfig.h"

namespace o2::its3::reco_workflow
{

framework::WorkflowSpec getWorkflow(bool useMC,
                                    bool doStag,
                                    its::TrackingMode::Type trmode,
                                    o2::gpu::gpudatatypes::DeviceType dtype,
                                    bool useGPUWorkflow,
                                    bool upstreamDigits,
                                    bool upstreamClusters,
                                    bool disableRootOutput,
                                    bool useGeom,
                                    int useTrig,
                                    bool overrideBeamPosition);
}

#endif
