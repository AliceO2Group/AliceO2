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

#include "TPCSimulation/ClustererTask.h"
#include "TPCSimulation/ClusterContainer.h"  // for ClusterContainer

#include "TObject.h"             // for TObject
#include "TClonesArray.h"        // for TClonesArray
#include "FairLogger.h"          // for LOG
#include "FairRootManager.h"     // for FairRootManager

ClassImp(o2::TPC::ClustererTask)

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
  , mClustersArray(nullptr)
  , mHwClustersArray(nullptr)
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

  mDigitsArray = dynamic_cast<TClonesArray *>(mgr->GetObject("TPCDigitMC"));
  if( !mDigitsArray ) {
    LOG(ERROR) << "TPC points not registered in the FairRootManager. Exiting ..." << FairLogger::endl;
    return kERROR;
  }

  if (mBoxClustererEnable) {
    mBoxClusterer = new BoxClusterer();
    mBoxClusterer->Init();
    
    // Register output container
    mClustersArray = new TClonesArray("o2::TPC::Cluster");
    mgr->Register("TPCCluster", "TPC", mClustersArray, kTRUE);
  }

  if (mHwClustererEnable) {
    mHwClusterer = new HwClusterer();
    mHwClusterer->setContinuousReadout(mIsContinuousReadout);
    mHwClusterer->Init();
// TODO: implement noise/pedestal objecta
//    mHwClusterer->setNoiseObject();
//    mHwClusterer->setPedestalObject();

    // Register output container
    mHwClustersArray = new TClonesArray("o2::TPC::Cluster");
    mgr->Register("TPCClusterHW", "TPC", mHwClustersArray, kTRUE);
  }

  return kSUCCESS;
}

//_____________________________________________________________________
void ClustererTask::Exec(Option_t *option)
{
  LOG(DEBUG) << "Running clusterization on new event with " << mDigitsArray->GetEntriesFast() << " digits" << FairLogger::endl;

  if (mBoxClustererEnable) {
    mClustersArray->Clear();
    ClusterContainer* clusters = mBoxClusterer->Process(mDigitsArray);
    clusters->FillOutputContainer(mClustersArray);
  }

  if (mHwClustererEnable) {
    mHwClustersArray->Clear();
    ClusterContainer* hwClusters = mHwClusterer->Process(mDigitsArray);
    hwClusters->FillOutputContainer(mHwClustersArray);
    LOG(DEBUG) << "Hw clusterer found " << mHwClustersArray->GetEntriesFast() << " clusters" << FairLogger::endl;
  }

}
