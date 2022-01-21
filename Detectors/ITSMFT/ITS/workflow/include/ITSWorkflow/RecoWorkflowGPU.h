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

#ifndef O2_ITS_RECOWORKFLOW_GPU_H
#define O2_ITS_RECOWORKFLOW_GPU_H

#include "Framework/WorkflowSpec.h"

#include "GPUO2Interface.h"
#include "GPUReconstruction.h"
#include "GPUChainITS.h"

namespace o2
{
namespace its
{
namespace reco_workflow_gpu
{

framework::WorkflowSpec getWorkflow(bool useMC, const std::string& trmode, o2::gpu::GPUDataTypes::DeviceType dType = o2::gpu::GPUDataTypes::DeviceType::CUDA,
                                    bool upstreamDigits = false, bool upstreamClusters = false, bool disableRootOutput = false, bool eencode = false);
}
} // namespace its
} // namespace o2
#endif
