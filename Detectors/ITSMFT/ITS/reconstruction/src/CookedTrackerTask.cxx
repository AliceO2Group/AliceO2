// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file  CookedTrackerTask.cxx
/// \brief Implementation of the ITS "Cooked Matrix" tracker task
/// \author iouri.belikov@cern.ch

#include "ITSReconstruction/CookedTrackerTask.h"
#include "ITSMFTReconstruction/Cluster.h"
#include "DetectorsBase/Utils.h"
#include "MathUtils/Cartesian3D.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/MCTruthContainer.h"

#include "FairLogger.h"      // for LOG
#include "FairRootManager.h" // for FairRootManager

ClassImp(o2::ITS::CookedTrackerTask)

using namespace o2::ITS;
using namespace o2::Base;
using namespace o2::Base::Utils;

//_____________________________________________________________________
CookedTrackerTask::CookedTrackerTask(Int_t n, Bool_t useMCTruth):FairTask("ITSCookedTrackerTask")
  ,mTracker(n){
  if (useMCTruth)
    mTrkLabels = new o2::dataformats::MCTruthContainer<o2::MCCompLabel>;
}

//_____________________________________________________________________
CookedTrackerTask::~CookedTrackerTask()
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

//_____________________________________________________________________
/// \brief Init function
/// Inititializes the tracker and connects input and output container
InitStatus CookedTrackerTask::Init()
{
  FairRootManager* mgr = FairRootManager::Instance();
  if (!mgr) {
    LOG(ERROR) << "Could not instantiate FairRootManager. Exiting ..." << FairLogger::endl;
    return kERROR;
  }

  mClustersArray = mgr->InitObjectAs<const std::vector<o2::ITSMFT::Cluster> *>("ITSCluster");
  if (!mClustersArray) {
    LOG(ERROR) << "ITS clusters not registered in the FairRootManager. Exiting ..." << FairLogger::endl;
    return kERROR;
  }

  // Register output container
  mgr->RegisterAny("ITSTrack", mTracksArray, kTRUE);

  // Register MC Truth container
  if (mTrkLabels) {
     mgr->RegisterAny("ITSTrackMCTruth", mTrkLabels, kTRUE);
     mClsLabels = mgr->InitObjectAs<const o2::dataformats::MCTruthContainer<o2::MCCompLabel> *>("ITSClusterMCTruth");
     if (!mClsLabels) {
        LOG(ERROR) << "ITS cluster labels not registered in the FairRootManager. Exiting ..."
                   << FairLogger::endl;
        return kERROR;
     }
  }
  
  GeometryTGeo* geom = GeometryTGeo::Instance();
  geom->fillMatrixCache( bit2Mask(TransformType::T2GRot) ); // make sure T2GRot matrices are loaded
  mTracker.setGeometry(geom);
  mTracker.setMCTruthContainers(mClsLabels, mTrkLabels);
  mTracker.setContinuousMode(mContinuousMode);
  
  return kSUCCESS;
}

//_____________________________________________________________________
void CookedTrackerTask::Exec(Option_t* option)
{
  if (mTracksArray) mTracksArray->clear();
  if (mTrkLabels) mTrkLabels->clear();
  LOG(DEBUG) << "Running digitization on new event" << FairLogger::endl;

  mTracker.process(*mClustersArray, *mTracksArray);
}
