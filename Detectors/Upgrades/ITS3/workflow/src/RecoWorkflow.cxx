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
#include "ITS3Workflow/TrackerSpec.h"
#include "ITS3Workflow/TrackWriterSpec.h"
#include "ITS3Workflow/DigitReaderSpec.h"
#include "Framework/Logger.h"

namespace o2::its3::reco_workflow
{

framework::WorkflowSpec getWorkflow(bool useMC, const std::string& trmode, o2::gpu::GPUDataTypes::DeviceType dtype,
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
    specs.emplace_back(o2::its3::getClusterWriterSpec(useMC));
  }

  if (trmode != "off") {
    specs.emplace_back(o2::its3::getTrackerSpec(useMC, useGeom, useTrig, trmode, overrideBeamPosition, dtype));

    if (!disableRootOutput) {
      specs.emplace_back(o2::its3::getTrackWriterSpec(useMC));
    }
  }

  return specs;
}

} // namespace o2::its3::reco_workflow
