//
//  ClustererTask.cxx
//  ALICEO2
//
//  Based on DigitizerTask
//
//

#include "TPCSimulation/ClustererTask.h"
#include "TPCSimulation/ClusterContainer.h"  // for ClusterContainer
#include "TPCSimulation/BoxClusterer.h"       // for Clusterer

#include "TObject.h"             // for TObject
#include "TClonesArray.h"        // for TClonesArray
#include "FairLogger.h"          // for LOG
#include "FairRootManager.h"     // for FairRootManager

ClassImp(AliceO2::TPC::ClustererTask)

using namespace AliceO2::TPC;

//_____________________________________________________________________
ClustererTask::ClustererTask():
  FairTask("TPCClustererTask"),
  mClusterer(nullptr),
  mDigitsArray(nullptr),
  mClustersArray(nullptr)
{
  mClusterer = new BoxClusterer();
}

//_____________________________________________________________________
ClustererTask::~ClustererTask()
{
  delete mClusterer;
  if (mClustersArray)
    delete mClustersArray;
}

//_____________________________________________________________________
/// \brief Init function
/// Inititializes the clusterer and connects input and output container
InitStatus ClustererTask::Init()
{
  FairRootManager *mgr = FairRootManager::Instance();
  if( !mgr ) {

    LOG(ERROR) << "Could not instantiate FairRootManager. Exiting ..." << FairLogger::endl;
    return kERROR;
  }

  mDigitsArray = dynamic_cast<TClonesArray *>(mgr->GetObject("TPCDigit"));
  if( !mDigitsArray ) {
    LOG(ERROR) << "TPC points not registered in the FairRootManager. Exiting ..." << FairLogger::endl;
    return kERROR;
  }

  // Register output container
//   fClustersArray = new TClonesArray("AliceO2::TPC::BoxCluster");
  mClustersArray = new TClonesArray("AliceO2::TPC::Cluster");
  mgr->Register("TPCCluster", "TPC", mClustersArray, kTRUE);

  mClusterer->Init();
  return kSUCCESS;
}

//_____________________________________________________________________
void ClustererTask::Exec(Option_t *option)
{
  mClustersArray->Clear();
  LOG(DEBUG) << "Running digitization on new event" << FairLogger::endl;

  ClusterContainer* clusters = mClusterer->Process(mDigitsArray);
  clusters->FillOutputContainer(mClustersArray);
}
