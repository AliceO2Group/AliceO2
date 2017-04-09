/// \file  CookedTrackerTask.cxx
/// \brief Implementation of the ITS "Cooked Matrix" tracker task
/// \author iouri.belikov@cern.ch

#include "ITSReconstruction/CookedTrackerTask.h"
#include "ITSReconstruction/Cluster.h"

#include "FairLogger.h"      // for LOG
#include "FairRootManager.h" // for FairRootManager
#include "TClonesArray.h"    // for TClonesArray

ClassImp(o2::ITS::CookedTrackerTask)

  using namespace o2::ITS;

//_____________________________________________________________________
CookedTrackerTask::CookedTrackerTask(Int_t n) : FairTask("ITSCookedTrackerTask"), mNumOfThreads(n), mClustersArray(nullptr), mTracksArray(nullptr) {}

//_____________________________________________________________________
CookedTrackerTask::~CookedTrackerTask()
{
  if (mTracksArray) {
    mTracksArray->Delete();
    delete mTracksArray;
  }
}

//_____________________________________________________________________
/// \brief Init function
/// Inititializes the tracker and connects input and output container
InitStatus CookedTrackerTask::Init()
{
  FairRootManager* mgr = FairRootManager::Instance();
  if (!mgr) {
    LOG(ERROR) << "Could not instantiate FairRootManager. Exiting ..." << FairLogger::endl;
    return kERROR;
  }

  mClustersArray = dynamic_cast<const TClonesArray*>(mgr->GetObject("ITSCluster"));
  if (!mClustersArray) {
    LOG(ERROR) << "ITS clusters not registered in the FairRootManager. Exiting ..." << FairLogger::endl;
    return kERROR;
  }

  // Register output container
  mTracksArray = new TClonesArray("o2::ITS::CookedTrack");
  mgr->Register("ITSTrack", "ITS", mTracksArray, kTRUE);

  mGeometry.Build(kTRUE);
  Cluster::setGeom(&mGeometry);

  mTracker.setNumberOfThreads(mNumOfThreads);
  
  return kSUCCESS;
}

//_____________________________________________________________________
void CookedTrackerTask::Exec(Option_t* option)
{
  mTracksArray->Clear();
  LOG(DEBUG) << "Running digitization on new event" << FairLogger::endl;

  mTracker.process(*mClustersArray, *mTracksArray);
}
