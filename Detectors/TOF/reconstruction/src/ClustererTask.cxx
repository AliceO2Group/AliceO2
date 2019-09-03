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
/// \brief Implementation of the TOF cluster finder task

#include "TOFReconstruction/ClustererTask.h"
#include "MathUtils/Cartesian3D.h"
#include "MathUtils/Utils.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/MCTruthContainer.h"

#include "FairLogger.h"      // for LOG
#include "FairRootManager.h" // for FairRootManager

ClassImp(o2::tof::ClustererTask);

using namespace o2::tof;

//_____________________________________________________________________
ClustererTask::ClustererTask(Bool_t useMCTruth) : FairTask("TOFClustererTask")
{
  if (useMCTruth)
    mClsLabels = new o2::dataformats::MCTruthContainer<o2::MCCompLabel>;
}

//_____________________________________________________________________
ClustererTask::~ClustererTask()
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
InitStatus ClustererTask::Init()
{
  FairRootManager* mgr = FairRootManager::Instance();
  if (!mgr) {
    LOG(ERROR) << "Could not instantiate FairRootManager. Exiting ...";
    return kERROR;
  }

  const std::vector<o2::tof::Digit>* arr = mgr->InitObjectAs<const std::vector<o2::tof::Digit>*>("TOFDigit");
  if (!arr) {
    LOG(ERROR) << "TOF digits not registered in the FairRootManager. Exiting ...";
    return kERROR;
  }
  mReader.setDigitArray(arr);

  if (mClsLabels) { // here we take the array of labels used for the digits
    mDigitMCTruth =
      mgr->InitObjectAs<const dataformats::MCTruthContainer<MCCompLabel>*>("TOFDigitMCTruth");
    if (!mDigitMCTruth) {
      LOG(ERROR) << "TOF MC Truth not registered in the FairRootManager. Exiting ...";
      return kERROR;
    }
  }

  // Register output container
  mgr->RegisterAny("TOFCluster", mClustersArray, kTRUE);

  // Register new MC Truth container --> here we will now associate to the clusters all labels that belonged to all digits that formed that cluster
  if (mClsLabels)
    mgr->RegisterAny("TOFClusterMCTruth", mClsLabels, kTRUE);

  mClusterer.setMCTruthContainer(mClsLabels);

  return kSUCCESS;
}

//_____________________________________________________________________
void ClustererTask::Exec(Option_t* option)
{
  if (mClustersArray)
    mClustersArray->clear();
  if (mClsLabels)
    mClsLabels->clear();
  LOG(DEBUG) << "Running clusterization on new event";

  mClusterer.process(mReader, *mClustersArray, mDigitMCTruth);
}
