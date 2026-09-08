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
///
/// \file CATrackerSpec.h
/// \brief ITS common-CA tracker DPL device with tracker-only outputs.

#ifndef O2_ITS_CA_WORKFLOW_CATRACKERSPEC_H_
#define O2_ITS_CA_WORKFLOW_CATRACKERSPEC_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include <gsl/span>

#include "DataFormatsITS/TrackITS.h"
#include "DataFormatsITSMFT/ROFRecord.h"
#include "DetectorsBase/GRPGeomHelper.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/Task.h"
#include "ITSMFTTracking/GenericTrackOutputAdapter.h"
#include "ITSMFTTracking/Configuration.h"
#include "ITSCAWorkflow/ConfigPreflight.h"
#include "ITSMFTTracking/ClusterDecoding.h"
#include "ITSCAWorkflow/PublicationAdapter.h"
#include "ITSMFTTracking/Tracker.h"
#include "ITSMFTTracking/TrackerTraits.h"
#include "ITSMFTTracking/WorkflowSession.h"
#include "ITSMFTTracking/IOUtils.h"
#include "ITSMFTTracking/ITSMFTDetectorDefinitions.h"
#include "ITSMFTTracking/ROFViews.h"
#include "ITSMFTTracking/ROFLookupTables.h"
#include "DataFormatsITSMFT/TopologyDictionary.h"
#include "SimulationDataFormat/MCCompLabel.h"

namespace o2::its::ca
{

using o2::itsmft::tracking::CATrackerPublicationAction;
using o2::itsmft::tracking::decideCATrackerPublicationAction;

/// ITS common-CA tracker DPL task. Owns the TimeFrame and composes the
/// workflow input/timing/publication edge with Tracker.
class CATrackerDPL : public o2::framework::Task
{
 public:
  CATrackerDPL(std::shared_ptr<o2::base::GRPGeomRequest> gr, WorkflowOptions options);
  ~CATrackerDPL() override = default;

  void init(framework::InitContext& ic) final;
  void run(framework::ProcessingContext& pc) final;
  void finaliseCCDB(framework::ConcreteDataMatcher& matcher, void* obj) final;

 private:
  void updateTimeDependentParams(framework::ProcessingContext& pc);
  void addTruthSeedingVertices(const o2::InteractionRecord& origin, gsl::span<const o2::itsmft::ROFRecord> rofs);
  void configureROFViews(gsl::span<const o2::itsmft::ROFRecord> rofs);
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
  WorkflowOptions mOptions;
  o2::itsmft::tracking::WorkflowSession mSession{"ITS", o2::itsmft::tracking::ITSNLayers};
  std::unique_ptr<o2::itsmft::tracking::TrackerTraits> mTrackerTraits;
  std::unique_ptr<o2::itsmft::tracking::Tracker> mTracker;
  std::unique_ptr<o2::itsmft::tracking::ClusterDecoder> mClusterDecoder;
  const o2::itsmft::TopologyDictionary* mDictionary = nullptr;
  o2::itsmft::tracking::ITSSharedClusterCompatibility mCompatibility;
  PublicationAdapter mPublication;
};

o2::framework::DataProcessorSpec getCATrackerSpec(const WorkflowOptions& options);

} // namespace o2::its::ca

#endif // O2_ITS_CA_WORKFLOW_CATRACKERSPEC_H_
