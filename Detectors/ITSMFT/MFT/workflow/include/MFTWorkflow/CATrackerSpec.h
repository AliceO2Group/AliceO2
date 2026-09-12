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

/// @file   CATrackerSpec.h

#ifndef O2_MFT_CATRACKERSPEC_H_
#define O2_MFT_CATRACKERSPEC_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include "DetectorsBase/GRPGeomHelper.h"
#include "CommonDataFormat/IRFrame.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/Task.h"
#include "ITSMFTTracking/GenericTrackOutputAdapter.h"
#include "ITSMFTTracking/Configuration.h"
#include "MFTWorkflow/CAWorkflowOptions.h"
#include "ITSMFTTracking/ClusterDecoding.h"
#include "ITSMFTTracking/IOUtils.h"
#include "ITSMFTTracking/ITSMFTDetectorDefinitions.h"
#include "ITSMFTTracking/WorkflowSession.h"
#include "ITSMFTTracking/Tracker.h"
#include "ITSMFTTracking/TrackerTraits.h"
#include "ITSMFTTracking/ROFViews.h"
#include "ITSMFTTracking/ROFLookupTables.h"
#include "DataFormatsITSMFT/TopologyDictionary.h"
#include "SimulationDataFormat/MCCompLabel.h"

namespace o2::mft
{

using o2::itsmft::tracking::CATrackerPublicationAction;
using o2::itsmft::tracking::decideCATrackerPublicationAction;

/// MFT CA tracker DPL task. Owns the TimeFrame and composes the workflow
/// input/timing/publication edge with Tracker.
class CATrackerDPL : public o2::framework::Task
{
 public:
  CATrackerDPL(std::shared_ptr<o2::base::GRPGeomRequest> gr,
               ca::TrackerOptions options);
  ~CATrackerDPL() override = default;

  void init(framework::InitContext& ic) final;
  void run(framework::ProcessingContext& pc) final;
  void finaliseCCDB(framework::ConcreteDataMatcher& matcher, void* obj) final;

 private:
  void updateTimeDependentParams(framework::ProcessingContext& pc);
  void configureROFViews(gsl::span<const o2::itsmft::ROFRecord> rofs,
                         gsl::span<const o2::dataformats::IRFrame> irFrames);
  void initialiseTracking();
  o2::itsmft::tracking::TrackingOutcome processTimeFrame(
    gsl::span<const o2::itsmft::ROFRecord> rofs,
    gsl::span<const o2::itsmft::CompClusterExt> clusters,
    gsl::span<const unsigned char> patterns,
    const o2::dataformats::MCTruthContainer<MCCompLabel>* labels);
  bool isActive() const noexcept { return mTracker != nullptr && mTracker->isConfiguredFor(mSession.frame); }

  std::shared_ptr<o2::base::GRPGeomRequest> mGGCCDBRequest;
  bool mUseMC = false;
  bool mTrackingInitialised = false;
  ca::TrackerOptions mOptions;
  o2::itsmft::tracking::WorkflowSession mSession{"MFT", o2::itsmft::tracking::MFTNLayers};
  std::unique_ptr<o2::itsmft::tracking::TrackerTraits> mTrackerTraits;
  std::unique_ptr<o2::itsmft::tracking::Tracker> mTracker;
  std::unique_ptr<o2::itsmft::tracking::ClusterDecoder> mClusterDecoder;
  const o2::itsmft::TopologyDictionary* mDictionary = nullptr;
  int mMFTROFrameLengthInBC = 0;
};

o2::framework::DataProcessorSpec getCATrackerSpec(const ca::TrackerOptions& options);

} // namespace o2::mft

#endif // O2_MFT_CATRACKERSPEC_H_
