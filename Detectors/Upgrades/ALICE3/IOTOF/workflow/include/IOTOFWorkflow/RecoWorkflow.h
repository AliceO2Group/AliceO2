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

#ifndef O2_TF3_RECOWORKFLOW_H
#define O2_TF3_RECOWORKFLOW_H

#include "Framework/WorkflowSpec.h"
#include <string>

namespace o2::iotof
{
namespace reco_workflow
{

o2::framework::WorkflowSpec getWorkflow(bool useMC,
                                        // const std::string& hitRecoConfig,
                                        bool upstreamDigits = false,
                                        bool upstreamClusters = false,
                                        bool disableRootOutput = false);
}

} // namespace o2::iotof

#endif
