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

#include "EC0Workflow/RecoWorkflow.h"

#include "EC0Workflow/DigitReaderSpec.h"
#include "EC0Workflow/ClustererSpec.h"
#include "EC0Workflow/ClusterWriterSpec.h"
#include "EC0Workflow/TrackerSpec.h"
#include "EC0Workflow/CookedTrackerSpec.h"
#include "EC0Workflow/TrackWriterSpec.h"
#include "EndCapsWorkflow/EntropyEncoderSpec.h"

namespace o2
{
namespace ecl
{

namespace reco_workflow
{

framework::WorkflowSpec getWorkflow(bool useMC, bool useCAtracker, o2::gpu::GPUDataTypes::DeviceType dtype,
                                    bool upstreamDigits, bool upstreamClusters, bool disableRootOutput,
                                    bool eencode)
{
  framework::WorkflowSpec specs;

  if (!(upstreamDigits || upstreamClusters)) {
    specs.emplace_back(o2::ecl::getDigitReaderSpec(useMC));
  }

  if (!upstreamClusters) {
    specs.emplace_back(o2::ecl::getClustererSpec(useMC));
  }
  if (!disableRootOutput) {
    specs.emplace_back(o2::ecl::getClusterWriterSpec(useMC));
  }
  if (useCAtracker) {
    specs.emplace_back(o2::ecl::getTrackerSpec(useMC, dtype));
  } else {
    specs.emplace_back(o2::ecl::getCookedTrackerSpec(useMC));
  }
  if (!disableRootOutput) {
    specs.emplace_back(o2::ecl::getTrackWriterSpec(useMC));
  }

  if (eencode) {
    specs.emplace_back(o2::endcaps::getEntropyEncoderSpec("EC0"));
  }
  return specs;
}

} // namespace reco_workflow
} // namespace ecl
} // namespace o2
