// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file ClustererTrack.h
/// \brief Cluster finding from digits (MFT)
/// \author bogdan.vulpescu@cern.ch
/// \date 03/05/2017

#include "MFTReconstruction/ClustererTask.h"
#include "MFTBase/Constants.h"
#include "MFTBase/Geometry.h"

#include "FairLogger.h"
#include "FairRootManager.h"

ClassImp(o2::MFT::ClustererTask);

using namespace o2::MFT;
using namespace o2::Base;
using namespace o2::utils;

//_____________________________________________________________________
ClustererTask::ClustererTask(bool useMC) : FairTask("MFTClustererTask"), mUseMCTruth(useMC)
{
}

//_____________________________________________________________________
ClustererTask::~ClustererTask()
{
  mClustersArray.clear();
  mClsLabels.clear();
}

//_____________________________________________________________________
InitStatus ClustererTask::Init()
{
  /// Inititializes the clusterer and connects input and output container
  FairRootManager* mgr = FairRootManager::Instance();
  if (!mgr) {
    LOG(ERROR) << "Could not instantiate FairRootManager. Exiting ..." << FairLogger::endl;
    return kERROR;
  }

  const auto arrDig = mgr->InitObjectAs<const std::vector<o2::ITSMFT::Digit>*>("MFTDigit");
  if (!arrDig) {
    LOG(ERROR) << "MFT digits are not registered in the FairRootManager. Exiting ..." << FairLogger::endl;
    return kERROR;
  }
  mReader.setDigitArray(arrDig);

  const auto arrDigLbl = mUseMCTruth ? mgr->InitObjectAs<const MCTruth*>("MFTDigitMCTruth") : nullptr;

  if (!arrDigLbl && mUseMCTruth) {
    LOG(WARNING) << "MFT digits labeals are not registered in the FairRootManager. Continue w/o MC truth ..."
                 << FairLogger::endl;
  }
  mClusterer.setDigitsMCTruthContainer(arrDigLbl);

  // Register output container
  mgr->RegisterAny("MFTCluster", mClustersArrayPtr, kTRUE);

  // Register output MC Truth container if there is an MC truth for digits
  if (arrDigLbl) {
    mgr->RegisterAny("MFTClusterMCTruth", mClsLabelsPtr, kTRUE);
    mClusterer.setClustersMCTruthContainer(mClsLabelsPtr);
  }

  GeometryTGeo* geom = GeometryTGeo::Instance();
  geom->fillMatrixCache(o2::utils::bit2Mask(o2::TransformType::T2L)); // make sure T2L matrices are loaded
  mGeometry = geom;
  mClusterer.setGeometry(geom);

  return kSUCCESS;
}

//_____________________________________________________________________
void ClustererTask::Exec(Option_t* option)
{
  mClustersArray.clear();
  mClsLabels.clear();

  LOG(DEBUG) << "Running clusterization on new event" << FairLogger::endl;

  mClusterer.process(mReader, mClustersArray);
}
