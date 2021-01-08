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

#include "FT0Simulation/DigitizerTask.h"
#include "MathUtils/Utils.h"

#include "FairLogger.h"      // for LOG
#include "FairRootManager.h" // for FairRootManager

using namespace o2::fit;

DigitizerTask::DigitizerTask() : FairTask("FT0DigitizerTask"), mDigitizer() {}
DigitizerTask::~DigitizerTask()
{
  if (mEventDigit)
    delete mEventDigit;
  mMCTruthArrayPtr->clear();
  delete mHitsArrayQED; // this special branch is managed by the task
}

/// Inititializes the digitizer and connects input and output container
InitStatus DigitizerTask::Init()
{
  FairRootManager* mgr = FairRootManager::Instance();
  if (!mgr) {
    LOG(ERROR) << "Could not instantiate FairRootManager. Exiting ...";
    return kERROR;
  }

  mHitsArray = mgr->InitObjectAs<const std::vector<o2::ft0::HitType>*>("FITHit");

  if (!mHitsArray) {
    LOG(ERROR) << "FIT hits not registered in the FairRootManager. Exiting ...";
    return kERROR;
  }

  mDigitizer.setMCLabels(mMCTruthArrayPtr);

  // Register output container
  mgr->RegisterAny("FT0Digit", mEventDigit, kTRUE);
  // mMCTruthArray = new typename std::remove_pointer<decltype(mMCTruthArray)>::type;
  mgr->RegisterAny("FITDigitMCTruth", mMCTruthArrayPtr, kTRUE);
  mDigitizer.init();
  return kSUCCESS;
}

//________________________________________________________
void DigitizerTask::Exec(Option_t* option)
{
  FairRootManager* mgr = FairRootManager::Instance();
  mMCTruthArrayPtr->clear();

  Float_t EventTime = mgr->GetEventTime();
  mDigitizer.setEventTime(EventTime);

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

  // is there QED backgroung provided? Fill QED slots up to the end of reserved ROFrames
  if (mQEDBranch) {
    Float_t digitROtime = 25; //ns
    processQEDBackground(digitROtime);
  }

  FairRootManager* mgr = FairRootManager::Instance();
  mgr->SetLastFill(kTRUE); /// necessary, otherwise the data is not written out

  mDigitizer.finish();
}
//________________________________________________________
void DigitizerTask::processQEDBackground(double tMax)
{
  // process QED time-slots until provided collision time (in ns)

  double tQEDNext = mLastQEDTimeNS + mQEDEntryTimeBinNS;

  while (tQEDNext < tMax) {
    mLastQEDTimeNS = tQEDNext;      // time used for current QED slot
    tQEDNext += mQEDEntryTimeBinNS; // prepare time for next QED slot
    if (++mLastQEDEntry >= mQEDBranch->GetEntries()) {
      mLastQEDEntry = 0; // wrapp if needed
    }
    mQEDBranch->GetEntry(mLastQEDEntry);
    mDigitizer.setEventTime(mLastQEDTimeNS);
    mDigitizer.setEventID(mQEDSourceID);

    mDigitizer.process(mHitsArrayQED, mEventDigit);
  }
}
