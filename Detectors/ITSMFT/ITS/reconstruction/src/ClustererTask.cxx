/// \file  ClustererTask.cxx
/// \brief Implementation of the ITS cluster finder task

#include "ITSReconstruction/ClustererTask.h"

#include "FairLogger.h"      // for LOG
#include "FairRootManager.h" // for FairRootManager
#include "TClonesArray.h"    // for TClonesArray

ClassImp(AliceO2::ITS::ClustererTask)

  using namespace AliceO2::ITS;

//_____________________________________________________________________
ClustererTask::ClustererTask() : FairTask("ITSClustererTask"), mDigitsArray(nullptr), mClustersArray(nullptr) {}

//_____________________________________________________________________
ClustererTask::~ClustererTask()
{
  if (mClustersArray) {
    mClustersArray->Delete();
    delete mClustersArray;
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

  mDigitsArray = dynamic_cast<TClonesArray*>(mgr->GetObject("ITSDigit"));
  if (!mDigitsArray) {
    LOG(ERROR) << "ITS points not registered in the FairRootManager. Exiting ..." << FairLogger::endl;
    return kERROR;
  }

  // Register output container
  mClustersArray = new TClonesArray("AliceO2::ITS::Cluster");
  mgr->Register("ITSCluster", "ITS", mClustersArray, kTRUE);

  mClusterer.init(kTRUE);

  return kSUCCESS;
}

//_____________________________________________________________________
void ClustererTask::Exec(Option_t* option)
{
  mClustersArray->Clear();
  LOG(DEBUG) << "Running digitization on new event" << FairLogger::endl;

  mClusterer.process(mDigitsArray, mClustersArray);
}
