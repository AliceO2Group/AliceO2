// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file  CollisionTimeRecoTask.cxx
/// \brief Implementation of the FIT reconstruction task

#include "FITReconstruction/CollisionTimeRecoTask.h"
#include "FITBase/Digit.h"
#include "FITReconstruction/RecPoints.h"

#include "FairLogger.h"      // for LOG
#include "FairRootManager.h" // for FairRootManager

ClassImp(o2::fit::CollisionTimeRecoTask);

using namespace o2::fit;
//_____________________________________________________________________
CollisionTimeRecoTask::CollisionTimeRecoTask() : FairTask("FITCollisionTimeRecoTask")
{
  //here
}

//_____________________________________________________________________
CollisionTimeRecoTask::~CollisionTimeRecoTask()
{
  if (mRecPoints) {
    delete mRecPoints;
  }
}

/// \brief Init function
/// Inititializes the clusterer and connects input and output container
InitStatus CollisionTimeRecoTask::Init()
{
  FairRootManager* mgr = FairRootManager::Instance();
  if (!mgr) {
    LOG(ERROR) << "Could not instantiate FairRootManager. Exiting ..." << FairLogger::endl;
    return kERROR;
  }

  //mDigitsArray = mgr->InitObjectAs<const std::vector<o2::fit::Digit>*>("FITDigit");
  mEventDigit = mgr->InitObjectAs<const Digit*>("FITDigit");
  //  if (!mDigitsArray) {
  //    LOG(ERROR) << "FIT digits not registered in the FairRootManager. Exiting ..." << FairLogger::endl;
  //    return kERROR;
  //  }

  // Register output container
  mgr->RegisterAny("FITRecPoints", mRecPoints, kTRUE);

  return kSUCCESS;
}
//_____________________________________________________________________
void CollisionTimeRecoTask::Exec(Option_t* option)
{
  LOG(DEBUG) << "Running reconstruction on new event" << FairLogger::endl;
  FairRootManager* mgr = FairRootManager::Instance();

  mRecPoints->FillFromDigits(*mEventDigit);
  mEventID++;
}
//________________________________________________________
void CollisionTimeRecoTask::FinishTask()
{
  // finalize digitization, if needed, flash remaining digits
  if (!mContinuous)
    return;
  FairRootManager* mgr = FairRootManager::Instance();
  mgr->SetLastFill(kTRUE); /// necessary, otherwise the data is not written out
}
