// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file DigitizerTask.cxx
/// \brief Implementation of the TOF digitizer task

#include "FDDSimulation/DigitizerTask.h"
#include "MathUtils/Utils.h"

#include "FairLogger.h"      // for LOG
#include "FairRootManager.h" // for FairRootManager

using namespace o2::fdd;

//________________________________________________________
DigitizerTask::DigitizerTask() : FairTask("FDDDigitizerTask"), mDigitizer() {}

//________________________________________________________
DigitizerTask::~DigitizerTask()
{
  if (mEventDigit)
    delete mEventDigit;
  mMCTruthArrayPtr->clear();
}

//________________________________________________________
InitStatus DigitizerTask::Init()
{
  FairRootManager* mgr = FairRootManager::Instance();
  if (!mgr) {
    LOG(ERROR) << "Could not instantiate FairRootManager. Exiting ..." << FairLogger::endl;
    return kERROR;
  }

  mHitsArray = mgr->InitObjectAs<const std::vector<o2::fdd::Hit>*>("FDDHit");

  if (!mHitsArray) {
    LOG(ERROR) << "FDD hits not registered in the FairRootManager. Exiting ..." << FairLogger::endl;
    return kERROR;
  }

  mDigitizer.setMCLabels(mMCTruthArrayPtr);

  // Register output container
  mgr->RegisterAny("FDDDigit", mEventDigit, kTRUE);
  mgr->RegisterAny("FDDDigitMCTruth", mMCTruthArrayPtr, kTRUE);
  mDigitizer.init();
  return kSUCCESS;
}

//________________________________________________________
void DigitizerTask::Exec(Option_t* option)
{
  FairRootManager* mgr = FairRootManager::Instance();
  mMCTruthArrayPtr->clear();

  Float_t EventTime = mgr->GetEventTime();
  mDigitizer.SetEventTime(EventTime);

  // the type of digitization is steered by the DigiParams object of the Digitizer
  LOG(DEBUG) << "Running digitization on new event " << mEventID << " from source " << mSourceID
             << " Event time " << EventTime;

  /// RS: ATTENTION: this is just a trick until we clarify how the hits from different source are
  /// provided and identified.
  mDigitizer.setEventID(mEventID);
  mDigitizer.setSrcID(mSourceID);

  mDigitizer.process(mHitsArray, mEventDigit);

  mEventID++;
}

//________________________________________________________
void DigitizerTask::FinishTask()
{
  // finalize digitization, if needed, flash remaining digits
  if (!mContinuous)
    return;

  FairRootManager* mgr = FairRootManager::Instance();
  mgr->SetLastFill(kTRUE); /// necessary, otherwise the data is not written out

  mDigitizer.finish();
}
