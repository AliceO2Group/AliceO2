/// \file  TrivialClustererTask.cxx
/// \brief Implementation of the ITS cluster finder task

#include "ITSReconstruction/TrivialClustererTask.h"

#include "FairLogger.h"      // for LOG
#include "FairRootManager.h" // for FairRootManager
#include "TClonesArray.h"    // for TClonesArray

ClassImp(o2::ITS::TrivialClustererTask)

using o2::ITSMFT::SegmentationPixel;
using namespace o2::ITS;

//_____________________________________________________________________
TrivialClustererTask::TrivialClustererTask() : FairTask("ITSTrivialClustererTask"), mDigitsArray(nullptr), mClustersArray(nullptr) {}

//_____________________________________________________________________
TrivialClustererTask::~TrivialClustererTask()
{
  if (mClustersArray) {
    mClustersArray->Delete();
    delete mClustersArray;
  }
}

//_____________________________________________________________________
/// \brief Init function
/// Inititializes the clusterer and connects input and output container
InitStatus TrivialClustererTask::Init()
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
  mClustersArray = new TClonesArray("o2::ITS::Cluster");
  mgr->Register("ITSCluster", "ITS", mClustersArray, kTRUE);

  mGeometry.Build(kTRUE);

  return kSUCCESS;
}

//_____________________________________________________________________
void TrivialClustererTask::Exec(Option_t* option)
{
  mClustersArray->Clear();
  LOG(DEBUG) << "Running digitization on new event" << FairLogger::endl;

  const SegmentationPixel* seg = (SegmentationPixel*)mGeometry.getSegmentationById(0);

  mTrivialClusterer.process(seg, mDigitsArray, mClustersArray);
}
