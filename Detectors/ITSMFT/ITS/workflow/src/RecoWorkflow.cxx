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
#include "ITSWorkflow/IRFrameWriterSpec.h"
#include "ITSWorkflow/TrackerSpec.h"
#include "ITSWorkflow/CookedTrackerSpec.h"
#include "ITSWorkflow/TrackWriterSpec.h"
#include "ITSMFTWorkflow/EntropyEncoderSpec.h"
#include "ITSMFTWorkflow/DigitReaderSpec.h"

namespace o2
{
namespace its
{

namespace reco_workflow
{

framework::WorkflowSpec getWorkflow(bool useMC, bool useCAtracker, const std::string& trmode, o2::gpu::GPUDataTypes::DeviceType dtype,
                                    bool upstreamDigits, bool upstreamClusters, bool disableRootOutput,
                                    bool eencode)
{
  framework::WorkflowSpec specs;

  if (!(upstreamDigits || upstreamClusters)) {
    specs.emplace_back(o2::itsmft::getITSDigitReaderSpec(useMC, false, "itsdigits.root"));
  }

  if (!upstreamClusters) {
    specs.emplace_back(o2::its::getClustererSpec(useMC));
  }
  if (!disableRootOutput) {
    specs.emplace_back(o2::its::getClusterWriterSpec(useMC));
    specs.emplace_back(o2::its::getTrackWriterSpec(useMC));
    specs.emplace_back(o2::its::getIRFrameWriterSpec());
  }
  if (useCAtracker) {
    specs.emplace_back(o2::its::getTrackerSpec(useMC, trmode, dtype));
  } else {
    specs.emplace_back(o2::its::getCookedTrackerSpec(useMC));
  }

  if (eencode) {
    specs.emplace_back(o2::itsmft::getEntropyEncoderSpec("ITS"));
  }
  return specs;
}

} // namespace reco_workflow
} // namespace its
} // namespace o2
