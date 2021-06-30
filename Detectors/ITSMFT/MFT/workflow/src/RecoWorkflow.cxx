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

#include <TTree.h>
#include "MFTWorkflow/RecoWorkflow.h"
#include "MFTWorkflow/ClustererSpec.h"
#include "MFTWorkflow/ClusterWriterSpec.h"
#include "MFTWorkflow/ClusterReaderSpec.h"
#include "MFTWorkflow/TrackerSpec.h"
#include "MFTWorkflow/TrackWriterSpec.h"
#include "ITSMFTWorkflow/DigitReaderSpec.h"

namespace o2
{
namespace mft
{

namespace reco_workflow
{

framework::WorkflowSpec getWorkflow(bool useMC, bool upstreamDigits, bool upstreamClusters, bool disableRootOutput)
{
  framework::WorkflowSpec specs;

  if (!(upstreamDigits || upstreamClusters)) {
    specs.emplace_back(o2::itsmft::getMFTDigitReaderSpec(useMC, false, "mftdigits.root"));
  }
  if (!upstreamClusters) {
    specs.emplace_back(o2::mft::getClustererSpec(useMC));
  }
  if (!disableRootOutput) {
    specs.emplace_back(o2::mft::getClusterWriterSpec(useMC));
  }
  specs.emplace_back(o2::mft::getTrackerSpec(useMC));
  if (!disableRootOutput) {
    specs.emplace_back(o2::mft::getTrackWriterSpec(useMC));
  }

  return specs;
}

} // namespace reco_workflow

} // namespace mft
} // namespace o2
