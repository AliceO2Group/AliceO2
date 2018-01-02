// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file  CA/TrackerTask.cxx
/// \brief Implementation of the ITS "Cellular Automaton" tracker task

#include "ITSReconstruction/CA/IOUtils.h"
#include "ITSReconstruction/CA/TrackerTask.h"
#include "ITSReconstruction/CA/Tracker.h"
#include "DataFormatsITSMFT/Cluster.h"
#include "MathUtils/Utils.h"
#include "MathUtils/Cartesian3D.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/MCTruthContainer.h"

#include "FairLogger.h"
#include "FairRootManager.h"

ClassImp(o2::ITS::CA::TrackerTask)

namespace o2
{
namespace ITS
{
namespace CA
{

TrackerTask::TrackerTask(bool useMCTruth) :
  FairTask{"ITSTrackerTask"},
  mTracker{}
{
  if (useMCTruth)
    mTrkLabels = new o2::dataformats::MCTruthContainer<o2::MCCompLabel>;
}


TrackerTask::~TrackerTask()
{
  if (mTracksArray) {
    mTracksArray->clear();
    delete mTracksArray;
  }
  if (mTrkLabels) {
    mTrkLabels->clear();
    delete mTrkLabels;
  }
}


/// \brief Init function
/// Inititializes the tracker and connects input and output container
InitStatus TrackerTask::Init()
{
  FairRootManager* mgr = FairRootManager::Instance();
  if (!mgr) {
    LOG(ERROR) << "Could not instantiate FairRootManager. Exiting." << FairLogger::endl;
    return kERROR;
  }

  mClustersArray = mgr->InitObjectAs<const std::vector<ITSMFT::Cluster> *>("ITSCluster");
  if (!mClustersArray) {
    LOG(ERROR) << "ITS clusters not registered in the FairRootManager. Exiting." << FairLogger::endl;
    return kERROR;
  }

  // Register output container
  mgr->RegisterAny("ITSTrack", mTracksArray, true);

  // Register MC Truth container
  if (mTrkLabels) {
    mgr->RegisterAny("ITSTrackMCTruth", mTrkLabels, true);
    mClsLabels = mgr->InitObjectAs<const dataformats::MCTruthContainer<MCCompLabel> *>("ITSClusterMCTruth");
    if (!mClsLabels) {
      LOG(ERROR) << "ITS cluster labels not registered in the FairRootManager. Exiting."
        << FairLogger::endl;
      return kERROR;
    }
  }

  GeometryTGeo* geom = GeometryTGeo::Instance();
  geom->fillMatrixCache(utils::bit2Mask(TransformType::T2GRot)); // make sure T2GRot matrices are loaded

  return kSUCCESS;
}


void TrackerTask::Exec(Option_t* option)
{
  if (mTracksArray) mTracksArray->clear();
  if (mTrkLabels) mTrkLabels->clear();

  IOUtils::loadEventData(mEvent,mClustersArray, mClsLabels);
  mTracker.clustersToTracks(mEvent);
}

}
}
}

