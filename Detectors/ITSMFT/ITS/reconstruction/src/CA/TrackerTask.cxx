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
/// \brief Implementation of the ITS CA tracker task

#include "ITSReconstruction/CA/TrackerTask.h"
#include "ITSMFTReconstruction/Cluster.h"
#include "DetectorsBase/Utils.h"
#include "MathUtils/Cartesian3D.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/MCTruthContainer.h"

#include "FairLogger.h"      // for LOG
#include "FairRootManager.h" // for FairRootManager

ClassImp(o2::ITS::CA::TrackerTask)

namespace o2 {
namespace ITS {
namespace CA {

TrackerTask::TrackerTask(bool useMCTruth) :
  FairTask{"ITS::CA::TrackerTask"},
  mGeometry{nullptr},
  mEvent{0,0.5f},
  mTracker{mEvent},
  mTrkLabels{useMCTruth ? new dataformats::MCTruthContainer<MCCompLabel>() : nullptr}
{
}

InitStatus TrackerTask::Init()
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
        LOG(ERROR) << "ITS cluster labels not registered in the FairRootManager. Exiting ..." << FairLogger::endl;
        return kERROR;
     }
  }

  mGeometry = GeometryTGeo::Instance();
  mGeometry->fillMatrixCache(Base::Utils::bit2Mask(Base::TransformType::T2GRot)); // make sure T2GRot matrices are loaded
  mEvent.setMCTruthContainers(mClsLabels);
  mTracker.setTrackMCTruthContainer(mTrkLabels);

  return kSUCCESS;
}

//_____________________________________________________________________
void TrackerTask::Exec(Option_t* option)
{
  if (mTracksArray) mTracksArray->clear();
  if (mTrkLabels) mTrkLabels->clear();

  LOG(DEBUG) << "Running tracking on new event" << FairLogger::endl;

  int numOfClusters = mClustersArray->size();
  if (numOfClusters == 0) {
    std::cout << "No clusters to load !" << std::endl;
    return;
  }

  mEvent.addPrimaryVertex(0.f,0.f,0.f);
  int currentLayer = 0;
  int offset = 0;
  for (int iCluster = 0; iCluster < numOfClusters; iCluster++) {
    const ITSMFT::Cluster& cluster = mClustersArray->at(iCluster);
    const int layerId = mGeometry->getLayer(cluster.getSensorID());
    if (layerId != currentLayer) {
      offset = iCluster;
      currentLayer = layerId;
    }
    float alphaRef = mGeometry->getSensorRefAlpha(cluster.getSensorID());
    mEvent.addTrackingFrameInfo(layerId,
      cluster.getX(),alphaRef,std::array<float, 2>{cluster.getY(),cluster.getZ()},
      std::array<float, 3>{cluster.getSigmaY2(), cluster.getSigmaYZ(), cluster.getSigmaZ2()}
    );
    auto xyz = cluster.getXYZGloRot(*mGeometry);
    float globalXYZ[3]{0.f};
    xyz.GetCoordinates(globalXYZ);
    mEvent.addCluster(layerId,iCluster - offset,globalXYZ[0],globalXYZ[1],globalXYZ[2],layerId);
  }
  mTracker.clustersToTracksVerbose();
  mEvent.clear();
}

}
}
}
