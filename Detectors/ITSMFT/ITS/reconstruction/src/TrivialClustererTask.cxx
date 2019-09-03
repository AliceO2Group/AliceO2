// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file  TrivialClustererTask.cxx
/// \brief Implementation of the ITS cluster finder task

#include "ITSReconstruction/TrivialClustererTask.h"
#include "ITSMFTBase/Digit.h"
#include "DataFormatsITSMFT/Cluster.h"
#include "MathUtils/Cartesian3D.h"
#include "MathUtils/Utils.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/MCTruthContainer.h"

#include "FairLogger.h"      // for LOG
#include "FairRootManager.h" // for FairRootManager

ClassImp(o2::its::TrivialClustererTask);

using namespace o2::its;
using namespace o2::base;
using namespace o2::utils;

//_____________________________________________________________________
TrivialClustererTask::TrivialClustererTask(Bool_t useMC) : FairTask("ITSTrivialClustererTask")
{
  if (useMC)
    mClsLabels = new o2::dataformats::MCTruthContainer<o2::MCCompLabel>;
}

//_____________________________________________________________________
TrivialClustererTask::~TrivialClustererTask()
{
  if (mClustersArray) {
    mClustersArray->clear();
    delete mClustersArray;
  }
  if (mClsLabels) {
    mClsLabels->clear();
    delete mClsLabels;
  }
}

//_____________________________________________________________________
/// \brief Init function
/// Inititializes the clusterer and connects input and output container
InitStatus TrivialClustererTask::Init()
{
  FairRootManager* mgr = FairRootManager::Instance();
  if (!mgr) {
    LOG(ERROR) << "Could not instantiate FairRootManager. Exiting ...";
    return kERROR;
  }

  mDigitsArray = mgr->InitObjectAs<const std::vector<o2::itsmft::Digit>*>("ITSDigit");
  if (!mDigitsArray) {
    LOG(ERROR) << "ITS points not registered in the FairRootManager. Exiting ...";
    return kERROR;
  }

  // Register output container
  mgr->RegisterAny("ITSCluster", mClustersArray, kTRUE);

  // Register MC Truth container
  if (mClsLabels)
    mgr->RegisterAny("ITSClusterMCTruth", mClsLabels, kTRUE);

  GeometryTGeo* geom = GeometryTGeo::Instance();
  geom->fillMatrixCache(o2::utils::bit2Mask(o2::TransformType::T2L)); // make sure T2L matrices are loaded
  mGeometry = geom;
  mTrivialClusterer.setGeometry(geom);
  mTrivialClusterer.setMCTruthContainer(mClsLabels);

  return kSUCCESS;
}

//_____________________________________________________________________
void TrivialClustererTask::Exec(Option_t* option)
{
  if (mClustersArray)
    mClustersArray->clear();
  if (mClsLabels)
    mClsLabels->clear();
  LOG(DEBUG) << "Running clusterization on new event";

  mTrivialClusterer.process(mDigitsArray, mClustersArray);
}
