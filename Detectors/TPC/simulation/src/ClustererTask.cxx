//
//  ClustererTask.cxx
//  ALICEO2
//
//  Based on DigitizerTask
//
//

#include "ClustererTask.h"
#include "ClusterContainer.h"  // for ClusterContainer
#include "BoxClusterer.h"       // for Clusterer

#include "TObject.h"             // for TObject
#include "TClonesArray.h"        // for TClonesArray
#include "FairLogger.h"          // for LOG
#include "FairRootManager.h"     // for FairRootManager

ClassImp(AliceO2::TPC::ClustererTask)

using namespace AliceO2::TPC;

//_____________________________________________________________________
ClustererTask::ClustererTask():
  FairTask("TPCClustererTask"),
  fClusterer(nullptr),
  fDigitsArray(nullptr),
  fClustersArray(nullptr)
{
  fClusterer = new BoxClusterer();
}

//_____________________________________________________________________
ClustererTask::~ClustererTask()
{
  delete fClusterer;
  if (fClustersArray) 
    delete fClustersArray;
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
  
  fDigitsArray = dynamic_cast<TClonesArray *>(mgr->GetObject("TPCDigit"));
  if( !fDigitsArray ) {
    LOG(ERROR) << "TPC points not registered in the FairRootManager. Exiting ..." << FairLogger::endl;
    return kERROR;
  }
  
  // Register output container
  fClustersArray = new TClonesArray("AliceO2::TPC::BoxCluster");
  mgr->Register("TPCCluster", "TPC", fClustersArray, kTRUE);      
  
  fClusterer->Init();
  return kSUCCESS;
}

//_____________________________________________________________________
void ClustererTask::Exec(Option_t *option)
{
  fClustersArray->Clear();
  LOG(DEBUG) << "Running digitization on new event" << FairLogger::endl;
  
  ClusterContainer* clusters = fClusterer->Process(fDigitsArray);
  clusters->FillOutputContainer(fClustersArray);
}
