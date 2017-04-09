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
ClustererTask::ClustererTask():
  FairTask("TPCClustererTask"),
  mBoxClustererEnable(false),
  mHwClustererEnable(false),
  mBoxClusterer(nullptr),
  mHwClusterer(nullptr),
  mDigitsArray(nullptr),
  mClustersArray(nullptr),
  mHwClustersArray(nullptr)
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
    mgr->Register("TPC_Cluster", "TPC", mClustersArray, kTRUE);
  }

  if (mHwClustererEnable) {
    mHwClusterer = new HwClusterer();
    mHwClusterer->Init();

    // Register output container
    mHwClustersArray = new TClonesArray("o2::TPC::Cluster");
    mgr->Register("TPC_HW_Cluster", "TPC", mHwClustersArray, kTRUE);
  }

  return kSUCCESS;
}

//_____________________________________________________________________
void ClustererTask::Exec(Option_t *option)
{
  LOG(DEBUG) << "Running clusterization on new event" << FairLogger::endl;

  if (mBoxClustererEnable) {
    mClustersArray->Clear();
    ClusterContainer* clusters = mBoxClusterer->Process(mDigitsArray);
    clusters->FillOutputContainer(mClustersArray);
  }

  if (mHwClustererEnable) {
    mHwClustersArray->Clear();
    ClusterContainer* hwClusters = mHwClusterer->Process(mDigitsArray);
    hwClusters->FillOutputContainer(mHwClustersArray);
  }

}
