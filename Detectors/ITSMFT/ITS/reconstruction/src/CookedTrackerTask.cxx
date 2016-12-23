/// \file  CookedTrackerTask.cxx
/// \brief Implementation of the ITS "Cooked Matrix" tracker task
/// \author iouri.belikov@cern.ch

#include "ITSReconstruction/CookedTrackerTask.h"
#include "ITSReconstruction/Cluster.h"

#include "FairLogger.h"      // for LOG
#include "FairRootManager.h" // for FairRootManager
#include "TClonesArray.h"    // for TClonesArray

ClassImp(AliceO2::ITS::CookedTrackerTask)

  using namespace AliceO2::ITS;

//_____________________________________________________________________
CookedTrackerTask::CookedTrackerTask() : FairTask("ITSCookedTrackerTask"), mClustersArray(nullptr), mTracksArray(nullptr) {}

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
  mTracksArray = new TClonesArray("AliceO2::ITS::CookedTrack");
  mgr->Register("ITSTrack", "ITS", mTracksArray, kTRUE);

  mGeometry.Build(kTRUE);
  Cluster::setGeom(&mGeometry);
  
  return kSUCCESS;
}

//_____________________________________________________________________
void CookedTrackerTask::Exec(Option_t* option)
{
  mTracksArray->Clear();
  LOG(DEBUG) << "Running digitization on new event" << FairLogger::endl;

  mTracker.process(*mClustersArray, *mTracksArray);
}
