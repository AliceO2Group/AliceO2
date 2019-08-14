// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file  ClustererTask.cxx
/// \brief Implementation of the PHOS cluster finder task

#include "FairLogger.h"      // for LOG
#include "FairRootManager.h" // for FairRootManager
#include "PHOSReconstruction/ClustererTask.h"
#include "PHOSReconstruction/Clusterer.h"
#include "PHOSReconstruction/Cluster.h"
#include "PHOSBase/Geometry.h"
#include "PHOSBase/Digit.h"

using namespace o2::phos;

ClassImp(ClustererTask);

//_____________________________________________________________________
ClustererTask::ClustererTask()
  : FairTask("PHOSClustererTask"), mClustersArray(nullptr), mDigitsArray(nullptr), mClusterer(nullptr)
{
}

//_____________________________________________________________________
ClustererTask::~ClustererTask()
{
  if (mClustersArray) {
    mClustersArray->clear();
    delete mClustersArray;
  }
  if (mClusterer) {
    delete mClusterer;
  }
}

//_____________________________________________________________________
/// \brief Init function
/// Inititializes the clusterer and connects input and output container
InitStatus ClustererTask::Init()
{
  FairRootManager* mgr = FairRootManager::Instance();
  if (!mgr) {
    LOG(ERROR) << "Could not instantiate FairRootManager. Exiting ...";
    return kERROR;
  }

  mDigitsArray = mgr->InitObjectAs<const std::vector<o2::phos::Digit>*>("PHSDigit");
  if (!mDigitsArray) {
    LOG(ERROR) << "PHOS digits not registered in the FairRootManager. Exiting ...";
    return kERROR;
  }

  // Register output container
  mgr->RegisterAny("PHSCluster", mClustersArray, kTRUE);

  mClusterer = new Clusterer();
  // TODO: set reco params/ref to RecoParam class???

  return kSUCCESS;
}

//_____________________________________________________________________
void ClustererTask::Exec(Option_t* option)
{
  if (mClustersArray)
    mClustersArray->clear();
  LOG(DEBUG) << "Running clusterization on new event";

  mClusterer->process(mDigitsArray, mClustersArray);
}
