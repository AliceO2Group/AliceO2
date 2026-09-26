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

/// @file CARecoWorkflow.cxx

#include "MFTWorkflow/CARecoWorkflow.h"

#include "GlobalTrackingWorkflowReaders/IRFrameReaderSpec.h"
#include "ITSMFTCAWriter/MFTCATrackWriterSpec.h"
#include "ITSMFTWorkflow/ClustererSpec.h"
#include "ITSMFTWorkflow/ClusterWriterSpec.h"
#include "ITSMFTWorkflow/DigitReaderSpec.h"
#include "MFTWorkflow/CATrackerSpec.h"
#include "MFTWorkflow/MFTAssessmentSpec.h"
#include "MFTWorkflow/TracksToRecordsSpec.h"

namespace o2::mft::ca_reco_workflow
{

framework::WorkflowSpec getWorkflow(const ca::WorkflowOptions& options)
{
  using namespace ca;
  framework::WorkflowSpec specs;
  const auto& tracker = options.tracker;
  const bool useGeom = tracker.geometry == GeometrySource::Full;
  const bool writeTracks = options.output == OutputPolicy::All || options.output == OutputPolicy::TracksAndClusterROFs;
  if (options.kind == WorkflowKind::Reconstruction) {
    if (options.input == InputStage::DigitsFile) {
      specs.emplace_back(o2::itsmft::getMFTDigitReaderSpec(tracker.useMC, options.staggering, false, true, "mftdigits.root"));
    }
    if (options.input != InputStage::UpstreamClusters) {
      specs.emplace_back(o2::itsmft::getMFTClustererSpec(tracker.useMC, options.staggering));
    }
    if (options.output != OutputPolicy::None) {
      specs.emplace_back(o2::itsmft::getMFTClusterWriterSpec(tracker.useMC, options.staggering, options.output != OutputPolicy::All));
    }
  }
  if (options.runTracking) {
    if (tracker.irFrames == IRFrameSource::File) {
      specs.emplace_back(o2::globaltracking::getIRFrameReaderSpec("ITS", 0, "its-irframe-reader", "o2_its_irframe.root"));
    }
    specs.emplace_back(o2::mft::getCATrackerSpec(tracker));
    if (writeTracks) {
      specs.emplace_back(o2::mft::getTrackWriterSpec(tracker.useMC, true));
    }
    if (options.assessment) {
      specs.emplace_back(o2::mft::getMFTAssessmentSpec(tracker.useMC, useGeom, options.processGenerated));
    }
    if (options.tracksToRecords) {
      specs.emplace_back(o2::mft::getTracksToRecordsSpec());
    }
  }
  return specs;
}

} // namespace o2::mft::ca_reco_workflow
