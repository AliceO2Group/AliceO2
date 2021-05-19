// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   RecoWorkflow.cxx

#include "ITS3Workflow/RecoWorkflow.h"
#include "ITS3Workflow/ClustererSpec.h"
#include "ITS3Workflow/ClusterWriterSpec.h"
// #include "ITSWorkflow/TrackerSpec.h"
// #include "ITSWorkflow/TrackWriterSpec.h"
// #include "ITSMFTWorkflow/EntropyEncoderSpec.h"
#include "ITS3Workflow/DigitReaderSpec.h"

namespace o2
{
namespace its3
{

namespace reco_workflow
{

framework::WorkflowSpec getWorkflow(bool useMC, const std::string& trmode, o2::gpu::GPUDataTypes::DeviceType dtype,
                                    bool upstreamDigits, bool upstreamClusters, bool disableRootOutput,
                                    bool)
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
  // specs.emplace_back(o2::its::getTrackerSpec(useMC, trmode, dtype));
  // if (!disableRootOutput) {
  //   specs.emplace_back(o2::its::getTrackWriterSpec(useMC));
  // }

  // if (eencode) {
  //   specs.emplace_back(o2::itsmft::getEntropyEncoderSpec("ITS"));
  // }
  return specs;
}

} // namespace reco_workflow
} // namespace its3
} // namespace o2
