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
/// \brief Implementation of the ITS cluster finder task

#include "ITSReconstruction/ClustererTask.h"
#include "DetectorsBase/Utils.h"
#include "MathUtils/Cartesian3D.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/MCTruthContainer.h"

#include "FairLogger.h"      // for LOG
#include "FairRootManager.h" // for FairRootManager

ClassImp(o2::ITS::ClustererTask)

using namespace o2::ITS;
using namespace o2::Base;
using namespace o2::Base::Utils;

//_____________________________________________________________________
ClustererTask::ClustererTask(Bool_t useMCTruth) : FairTask("ITSClustererTask") {
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
    LOG(ERROR) << "Could not instantiate FairRootManager. Exiting ..." << FairLogger::endl;
    return kERROR;
  }

  const std::vector<o2::ITSMFT::Digit> *arr =
    mgr->InitObjectAs<const std::vector<o2::ITSMFT::Digit> *>("ITSDigit");
  if (!arr) {
    LOG(ERROR)<<"ITS digits not registered in the FairRootManager. Exiting ..."<<FairLogger::endl;
    return kERROR;
  }
  mReader.setDigitArray(arr);
  
  // Register output container
  mgr->RegisterAny("ITSCluster", mClustersArray, kTRUE);

  // Register MC Truth container
  if (mClsLabels)
  mgr->Register("ITSClusterMCTruth", "ITS", mClsLabels, kTRUE);

  GeometryTGeo* geom = GeometryTGeo::Instance();
  geom->fillMatrixCache( bit2Mask(TransformType::T2L) ); // make sure T2L matrices are loaded
  mGeometry = geom;
  mClusterer.setGeometry(geom);
  mClusterer.setMCTruthContainer(mClsLabels);
  
  return kSUCCESS;
}

//_____________________________________________________________________
void ClustererTask::Exec(Option_t* option)
{
  if (mClustersArray) mClustersArray->clear();
  if (mClsLabels)  mClsLabels->clear();
  LOG(DEBUG) << "Running clusterization on new event" << FairLogger::endl;

  mClusterer.process(mReader, *mClustersArray);
}
