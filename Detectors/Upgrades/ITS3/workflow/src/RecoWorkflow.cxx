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
#include "ITS3Workflow/ClusterWriterSpec.h"
#include "ITSWorkflow/TrackerSpec.h"
#include "ITSWorkflow/TrackWriterSpec.h"
#include "ITS3Workflow/DigitReaderSpec.h"

namespace o2
{
namespace its3
{

namespace reco_workflow
{

framework::WorkflowSpec getWorkflow(bool useMC, const std::string& trmode, o2::gpu::GPUDataTypes::DeviceType dtype,
                                    bool upstreamDigits, bool upstreamClusters, bool disableRootOutput, int useTrig)
{
  framework::WorkflowSpec specs;

  if (!(upstreamDigits || upstreamClusters)) {
    specs.emplace_back(o2::its3::getITS3DigitReaderSpec(useMC, false, "it3digits.root"));
  }

  if (!upstreamClusters) {
    specs.emplace_back(o2::its3::getClustererSpec(useMC));
  }

  if (!disableRootOutput) {
    specs.emplace_back(o2::its3::getClusterWriterSpec(useMC));
  }

  // specs.emplace_back(o2::its::getTrackerSpec(useMC, useTrig, trmode, dtype));
  // if (!disableRootOutput) {
  //   specs.emplace_back(o2::its::getTrackWriterSpec(useMC));
  // }

  return specs;
}

} // namespace reco_workflow
} // namespace its3
} // namespace o2
