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

#include "TRKWorkflow/RecoWorkflow.h"
#include "TRKWorkflow/ClustererSpec.h"
#include "TRKWorkflow/ClusterWriterSpec.h"
#include "TRKWorkflow/DigitReaderSpec.h"
#include "TRKWorkflow/TrackerSpec.h"
#include "TRKWorkflow/TrackWriterSpec.h"
#include "Framework/CCDBParamSpec.h"

#include <string>

namespace o2::trk::reco_workflow
{

framework::WorkflowSpec getWorkflow(bool useMC,
                                    const std::string& hitRecoConfig,
                                    bool upstreamDigits,
                                    bool upstreamClusters,
                                    bool disableRootOutput,
                                    bool useGPUWF,
                                    o2::gpu::gpudatatypes::DeviceType dtype)
{
  framework::WorkflowSpec specs;

  if (!(upstreamDigits || upstreamClusters)) {
    specs.emplace_back(o2::trk::getTRKDigitReaderSpec(useMC, false, "trkdigits.root"));
  }
  if (!upstreamClusters) {
    specs.emplace_back(o2::trk::getClustererSpec(useMC));
  }

  if (!disableRootOutput) {
    specs.emplace_back(o2::trk::getClusterWriterSpec(useMC));
  }

  if (!hitRecoConfig.empty()) {
    LOGP(info, "Using hit reco config from file {}", hitRecoConfig);
    specs.emplace_back(o2::trk::getTrackerSpec(useMC, hitRecoConfig, dtype));
    if (!disableRootOutput) {
      specs.emplace_back(o2::trk::getTrackWriterSpec(useMC));
    }
  }

  return specs;
}

} // namespace o2::trk::reco_workflow
