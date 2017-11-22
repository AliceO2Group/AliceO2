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
#include "TPCReconstruction/Cluster.h"
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
  , mEventCount(0)
  , mBoxClusterer(nullptr)
  , mHwClusterer(nullptr)
  , mDigitsArray(nullptr)
  , mDigitMCTruthArray(nullptr)
  , mClustersArray(nullptr)
  , mHwClustersArray(nullptr)
  , mClustersMCTruthArray(nullptr)
  , mHwClustersMCTruthArray(nullptr)
{
}

//_____________________________________________________________________
ClustererTask::~ClustererTask()
{
  LOG(DEBUG) << "Enter Destructor of ClustererTask" << FairLogger::endl;

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
    LOG(ERROR) << "TPC MC Truth not registered in the FairRootManager. Exiting ..." << FairLogger::endl;
    return kERROR;
  }

  if (mBoxClustererEnable) {

    // Create and register output container
    mClustersArray = new std::vector<o2::TPC::Cluster>;
    mgr->RegisterAny("TPCCluster", mClustersArray, kTRUE);

    // Register MC Truth output container
    mClustersMCTruthArray = std::make_unique<MCLabelContainer>();
    // a trick to register the unique pointer with FairRootManager
    static auto tmp = mClustersMCTruthArray.get();
    mgr->RegisterAny("TPCClusterMCTruth", tmp, kTRUE);

    // create clusterer and pass output pointer
    mBoxClusterer = std::make_unique<BoxClusterer>(mClustersArray);
  }

  if (mHwClustererEnable) {
    // Register output container
    mHwClustersArray = new std::vector<o2::TPC::Cluster>;
    mgr->RegisterAny("TPCClusterHW", mHwClustersArray, kTRUE);

    // Register MC Truth output container
    mHwClustersMCTruthArray = std::make_unique<MCLabelContainer>();
    // a trick to register the unique pointer with FairRootManager
    static auto tmp = mHwClustersMCTruthArray.get();
    mgr->RegisterAny("TPCClusterHWMCTruth", tmp, kTRUE);

     // create clusterer and pass output pointer
    mHwClusterer = std::make_unique<HwClusterer>(mHwClustersArray,mHwClustersMCTruthArray.get());//,0,359);
    mHwClusterer->setContinuousReadout(mIsContinuousReadout);
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
    if(mClustersMCTruthArray) mClustersMCTruthArray->clear();
    mBoxClusterer->Process(*mDigitsArray,mDigitMCTruthArray,mEventCount);
    LOG(DEBUG) << "Box clusterer found " << mClustersArray->size() << " clusters" << FairLogger::endl << FairLogger::endl;
  }

  if (mHwClustererEnable) {
    if(mHwClustersArray) mHwClustersArray->clear();
    if(mHwClustersMCTruthArray) mHwClustersMCTruthArray->clear();
    mHwClusterer->Process(*mDigitsArray,mDigitMCTruthArray,mEventCount);
    LOG(DEBUG) << "Hw clusterer found " << mHwClustersArray->size() << " clusters" << FairLogger::endl << FairLogger::endl;
  }
  ++mEventCount;
}
