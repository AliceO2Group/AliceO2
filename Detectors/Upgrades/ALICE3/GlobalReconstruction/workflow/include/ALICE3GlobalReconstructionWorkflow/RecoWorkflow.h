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

#ifndef O2_ALICE3_GLOBALRECONSTRUCTION_RECOWORKFLOW_H
#define O2_ALICE3_GLOBALRECONSTRUCTION_RECOWORKFLOW_H

#include "Framework/WorkflowSpec.h"
#include "GPUDataTypesConfig.h"
#include <string>

namespace o2::trk::global_reco_workflow
{

o2::framework::WorkflowSpec getWorkflow(bool useMC,
                                        const std::string& hitRecoConfig,
                                        const std::string& clusterRecoConfig,
                                        bool disableRootOutput = false,
                                        o2::gpu::gpudatatypes::DeviceType dType = o2::gpu::gpudatatypes::DeviceType::CPU);

} // namespace o2::trk::global_reco_workflow

#endif
