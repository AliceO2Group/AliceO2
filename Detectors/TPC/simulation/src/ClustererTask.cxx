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

ClassImp(AliceO2::TPC::ClustererTask)

using namespace AliceO2::TPC;

//_____________________________________________________________________
ClustererTask::ClustererTask():
  FairTask("TPCClustererTask"),
  mBoxClustererEnable(false),
  mHwClustererEnable(false),
  fBoxClusterer(nullptr),
  fHwClusterer(nullptr),
  fDigitsArray(nullptr),
  fClustersArray(nullptr),
  fHwClustersArray(nullptr)
{
}

//_____________________________________________________________________
ClustererTask::~ClustererTask()
{
  LOG(DEBUG) << "Enter Destructor of ClustererTask" << FairLogger::endl;

  if (mBoxClustererEnable)  delete fBoxClusterer;
  if (mHwClustererEnable)   delete fHwClusterer;

  if (fClustersArray)
    delete fClustersArray;
  if (fHwClustersArray)
    delete fHwClustersArray;
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

  fDigitsArray = dynamic_cast<TClonesArray *>(mgr->GetObject("TPCDigit"));
  if( !fDigitsArray ) {
    LOG(ERROR) << "TPC points not registered in the FairRootManager. Exiting ..." << FairLogger::endl;
    return kERROR;
  }

  if (mBoxClustererEnable) {
    fBoxClusterer = new BoxClusterer();
    fBoxClusterer->Init();
    
    // Register output container
    fClustersArray = new TClonesArray("AliceO2::TPC::Cluster");
    mgr->Register("TPC_Cluster", "TPC", fClustersArray, kTRUE);
  }

  if (mHwClustererEnable) {
    fHwClusterer = new HwClusterer();
    fHwClusterer->Init();

    // Register output container
    fHwClustersArray = new TClonesArray("AliceO2::TPC::Cluster");
    mgr->Register("TPC_HW_Cluster", "TPC", fHwClustersArray, kTRUE);
  }

  return kSUCCESS;
}

//_____________________________________________________________________
void ClustererTask::Exec(Option_t *option)
{
  LOG(DEBUG) << "Running clusterization on new event" << FairLogger::endl;

  if (mBoxClustererEnable) {
    fClustersArray->Clear();
    ClusterContainer* clusters = fBoxClusterer->Process(fDigitsArray);
    clusters->FillOutputContainer(fClustersArray);
  }

  if (mHwClustererEnable) {
    fHwClustersArray->Clear();
    ClusterContainer* hwClusters = fHwClusterer->Process(fDigitsArray);
    hwClusters->FillOutputContainer(fHwClustersArray);
  }

}
