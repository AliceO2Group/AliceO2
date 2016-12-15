//
//  ClustererTask.cxx
//  ALICEO2
//

#include "ITSReconstruction/ClustererTask.h"
#include "ITSReconstruction/Cluster.h"
#include "ITSBase/Digit.h"
#include "ITSBase/SegmentationPixel.h"

#include "TClonesArray.h"        // for TClonesArray
#include "FairLogger.h"          // for LOG
#include "FairRootManager.h"     // for FairRootManager

ClassImp(AliceO2::ITS::ClustererTask)

using namespace AliceO2::ITS;

//_____________________________________________________________________
ClustererTask::ClustererTask():
  FairTask("ITSClustererTask"),
  fDigitsArray(nullptr),
  fClustersArray(nullptr)
{
}

//_____________________________________________________________________
ClustererTask::~ClustererTask()
{
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

  fDigitsArray = dynamic_cast<TClonesArray *>(mgr->GetObject("ITSDigit"));
  if( !fDigitsArray ) {
    LOG(ERROR) << "ITS points not registered in the FairRootManager. Exiting ..." << FairLogger::endl;
    return kERROR;
  }

  // Register output container
  fClustersArray = new TClonesArray("AliceO2::ITS::Cluster");
  mgr->Register("ITSCluster", "ITS", fClustersArray, kTRUE);

  fClusterer.Init(kTRUE);
  
  return kSUCCESS;
}

//_____________________________________________________________________
void ClustererTask::Exec(Option_t *option)
{
  fClustersArray->Clear();
  LOG(DEBUG) << "Running digitization on new event" << FairLogger::endl;

  fClusterer.Process(fDigitsArray,fClustersArray);

}
