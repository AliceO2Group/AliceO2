// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

//
//  ClustererTask.cxx
//  ALICEO2
//
//  Based on DigitizerTask
//
//

#include "TPCReconstruction/ClustererTask.h"
#include "TPCReconstruction/ClusterContainer.h"  // for ClusterContainer
#include "TPCBase/Digit.h"
#include "FairLogger.h"          // for LOG
#include "FairRootManager.h"     // for FairRootManager

ClassImp(o2::TPC::ClustererTask);

using namespace o2::TPC;

//_____________________________________________________________________
ClustererTask::ClustererTask()
  : FairTask("TPCClustererTask")
  , mBoxClustererEnable(false)
  , mHwClustererEnable(false)
  , mIsContinuousReadout(true)
  , mBoxClusterer(nullptr)
  , mHwClusterer(nullptr)
  , mDigitsArray(nullptr)
  , mDigitMCTruthArray(nullptr)
  , mClustersArray(nullptr)
  , mHwClustersArray(nullptr)
  , mClustersMCTruthArray()
  , mHwClustersMCTruthArray()
{
}

//_____________________________________________________________________
ClustererTask::~ClustererTask()
{
  LOG(DEBUG) << "Enter Destructor of ClustererTask" << FairLogger::endl;

  if (mBoxClustererEnable)  delete mBoxClusterer;
  if (mHwClustererEnable)   delete mHwClusterer;

  if (mClustersArray)
    delete mClustersArray;
  if (mHwClustersArray)
    delete mHwClustersArray;
}

//_____________________________________________________________________
/// \brief Init function
/// Inititializes the clusterer and connects input and output container
InitStatus ClustererTask::Init()
{
  LOG(DEBUG) << "Enter Initializer of ClustererTask" << FairLogger::endl;

  FairRootManager *mgr = FairRootManager::Instance();
  if( !mgr ) {

    LOG(ERROR) << "Could not instantiate FairRootManager. Exiting ..." << FairLogger::endl;
    return kERROR;
  }

  mDigitsArray = mgr->InitObjectAs<decltype(mDigitsArray)>("TPCDigit");
  if( !mDigitsArray ) {
    LOG(ERROR) << "TPC points not registered in the FairRootManager. Exiting ..." << FairLogger::endl;
    return kERROR;
  }

  mDigitMCTruthArray = mgr->InitObjectAs<decltype(mDigitMCTruthArray)>("TPCDigitMCTruth");
  if( !mDigitMCTruthArray ) {
    LOG(ERROR) << "TPC MC truth not registered in the FairRootManager. Exiting ..." << FairLogger::endl;
    return kERROR;
  }

  if (mBoxClustererEnable) {

    // Create and register output container
    mClustersArray = new std::vector<o2::TPC::Cluster>;
    mgr->RegisterAny("TPCCluster", mClustersArray, kTRUE);

    mgr->Register("TPCClusterMCTruth", "TPC", &mClustersMCTruthArray, kTRUE);

    // create clusterer and pass output pointer
    mBoxClusterer = new BoxClusterer(mClustersArray);
    mBoxClusterer->Init();
  }

  if (mHwClustererEnable) {
    // Register output container
    mHwClustersArray = new std::vector<o2::TPC::HwCluster>;
    mgr->RegisterAny("TPCClusterHW", mHwClustersArray, kTRUE);

    mgr->Register("TPCClusterHWMCTruth", "TPC", &mHwClustersMCTruthArray, kTRUE);

     // create clusterer and pass output pointer
    mHwClusterer = new HwClusterer(mHwClustersArray);
    mHwClusterer->setContinuousReadout(mIsContinuousReadout);
    mHwClusterer->Init();
// TODO: implement noise/pedestal objecta
//    mHwClusterer->setNoiseObject();
//    mHwClusterer->setPedestalObject();
  }

  return kSUCCESS;
}

//_____________________________________________________________________
void ClustererTask::Exec(Option_t *option)
{
  LOG(DEBUG) << "Running clusterization on new event with " << mDigitsArray->size() << " digits" << FairLogger::endl;

  if (mBoxClustererEnable) {
    mClustersArray->clear();
    mBoxClusterer->Process(*mDigitsArray,mDigitMCTruthArray,mClustersMCTruthArray);
  }

  if (mHwClustererEnable) {
    mHwClustersArray->clear();
    mHwClustersMCTruthArray.clear();
    mHwClusterer->Process(*mDigitsArray,mDigitMCTruthArray,mHwClustersMCTruthArray);
    LOG(DEBUG) << "Hw clusterer found " << mHwClustersArray->size() << " clusters" << FairLogger::endl;
  }
}
