// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file  TrackerTask.cxx
/// \brief Task driving the track finding from MFT clusters
/// \author bogdan.vulpescu@cern.ch
/// \date Feb. 7, 2018

#include "MFTReconstruction/TrackerTask.h"
#include "DataFormatsITSMFT/Cluster.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/MCTruthContainer.h"

ClassImp(o2::mft::TrackerTask);

using namespace o2::mft;
using namespace o2::base;

//_____________________________________________________________________________
TrackerTask::TrackerTask(Int_t n, Bool_t useMCTruth) : FairTask("MFTTrackerTask"), mTracker(n)
{

  if (useMCTruth) {
    mTrkLabels = new o2::dataformats::MCTruthContainer<o2::MCCompLabel>;
  }
}

//_____________________________________________________________________________
TrackerTask::~TrackerTask()
{

  if (mTracksArray) {
    delete mTracksArray;
  }
  if (mTrkLabels) {
    delete mTrkLabels;
  }
}

//_____________________________________________________________________________
InitStatus TrackerTask::Init()
{

  FairRootManager* mgr = FairRootManager::Instance();
  if (!mgr) {
    LOG(ERROR) << "Could not instantiate FairRootManager. Exiting ...";
    return kERROR;
  }

  mClustersArray = mgr->InitObjectAs<const std::vector<o2::itsmft::Cluster>*>("MFTCluster");
  if (!mClustersArray) {
    LOG(ERROR) << "MFT clusters not registered in the FairRootManager. Exiting ...";
    return kERROR;
  }

  // Register output container
  if (mTracksArray) {
    mgr->RegisterAny("MFTTrack", mTracksArray, kTRUE);
  }

  // Register MC Truth container
  if (mTrkLabels) {
    mgr->RegisterAny("MFTTrackMCTruth", mTrkLabels, kTRUE);
    mClsLabels = mgr->InitObjectAs<const o2::dataformats::MCTruthContainer<o2::MCCompLabel>*>("MFTClusterMCTruth");
    if (!mClsLabels) {
      LOG(ERROR) << "MFT cluster labels not registered in the FairRootManager. Exiting ...";
      return kERROR;
    }
  }

  GeometryTGeo* geom = GeometryTGeo::Instance();
  geom->fillMatrixCache(o2::utils::bit2Mask(o2::TransformType::L2G));
  // geom->fillMatrixCache(o2::utils::bit2Mask(o2::TransformType::T2L, o2::TransformType::T2G, o2::TransformType::L2G));
  mTracker.setGeometry(geom);
  mTracker.setMCTruthContainers(mClsLabels, mTrkLabels);
  mTracker.setContinuousMode(mContinuousMode);

  return kSUCCESS;
}

//_____________________________________________________________________________
void TrackerTask::Exec(Option_t* option)
{

  if (mTracksArray) {
    mTracksArray->clear();
  }
  if (mTrkLabels) {
    mTrkLabels->clear();
  }
  LOG(DEBUG) << "Running digitization on new event";

  mTracker.process(*mClustersArray, *mTracksArray);
}
