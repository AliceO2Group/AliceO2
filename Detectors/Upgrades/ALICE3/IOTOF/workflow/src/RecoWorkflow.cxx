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

#include "IOTOFWorkflow/RecoWorkflow.h"
#include "IOTOFWorkflow/DigitReaderSpec.h"
#include "IOTOFWorkflow/ClustererSpec.h"
#include "IOTOFWorkflow/ClusterWriterSpec.h"
#include "Framework/CCDBParamSpec.h"

#include <string>

namespace o2::iotof::reco_workflow
{

framework::WorkflowSpec getWorkflow(bool useMC,
                                    // const std::string& hitRecoConfig,
                                    bool upstreamDigits,
                                    bool upstreamClusters,
                                    bool disableRootOutput)
{
  LOG(info) << "[RecoWorkflow] ENTERING IOTOF RecoWorkflow.cxx";
  framework::WorkflowSpec specs;

  LOG(info) << "[RecoWorkflow] useMC: " << useMC;
  LOG(info) << "[RecoWorkflow] upstreamDigits: " << upstreamDigits;
  LOG(info) << "[RecoWorkflow] upstreamClusters: " << upstreamClusters;
  LOG(info) << "[RecoWorkflow] disableRootOutput: " << disableRootOutput;
  if (!(upstreamDigits || upstreamClusters)) {
    LOG(info) << "[RecoWorkflow] Adding DigitReaderSpec to workflow";
    specs.emplace_back(o2::iotof::getIOTOFDigitReaderSpec(useMC, false, "tf3digits.root"));
  }
  if (!upstreamClusters) {
    LOG(info) << "[RecoWorkflow] Adding ClustererSpec to workflow";
    specs.emplace_back(o2::iotof::getIOTOFClustererSpec(useMC));
  }

  if (!disableRootOutput) {
    LOG(info) << "[RecoWorkflow] Adding ClusterWriterSpec to workflow";
    specs.emplace_back(o2::iotof::getIOTOFClusterWriterSpec(useMC, false));
  }

  LOG(info) << "[RecoWorkflow] IOTOF RecoWorkflow.cxx completed, starting execution of workflow with " << specs.size() << " specifications";
  return specs;
}

} // namespace o2::iotof::reco_workflow
