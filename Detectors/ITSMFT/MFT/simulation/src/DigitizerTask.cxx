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
/// \brief Task driving the convertion from Hit to Digit
/// \author bogdan.vulpescu@cern.ch
/// \date 03/05/2017

#include "MFTSimulation/DigitizerTask.h"
#include "ITSMFTSimulation/Hit.h"
#include "MFTBase/GeometryTGeo.h"
#include "MathUtils/Utils.h"

#include "FairLogger.h"      // for LOG
#include "FairRootManager.h" // for FairRootManager

ClassImp(o2::mft::DigitizerTask);

using namespace o2::mft;
using namespace o2::detectors;
using namespace o2::utils;

using o2::itsmft::DigiParams;

//_____________________________________________________________________________
DigitizerTask::DigitizerTask() : FairTask("MFTDigitizerTask"), mDigitizer() {}
//_____________________________________________________________________________
DigitizerTask::~DigitizerTask()
{
  mDigitsArray.clear();
  mMCTruthArray.clear();
  delete mHitsArrayQED; // this special branch is managed by the task
}

/// \brief Init function
///
/// Inititializes the digitizer and connects input and output container
InitStatus DigitizerTask::Init()
{
  FairRootManager* mgr = FairRootManager::Instance();
  if (!mgr) {
    LOG(ERROR) << "Could not instantiate FairRootManager. Exiting ...";
    return kERROR;
  }

  mHitsArray = mgr->InitObjectAs<const std::vector<o2::itsmft::Hit>*>("MFTHit");
  if (!mHitsArray) {
    LOG(ERROR) << "MFT hits not registered in the FairRootManager. Exiting ...";
    return kERROR;
  }

  // Register output container
  mgr->RegisterAny("MFTDigit", mDigitsArrayPtr, kTRUE);
  mgr->RegisterAny("MFTDigitMCTruth", mMCTruthArrayPtr, kTRUE);

  GeometryTGeo* geom = GeometryTGeo::Instance();
  geom->fillMatrixCache(o2::utils::bit2Mask(o2::TransformType::L2G)); // make sure L2G matrices are loaded
  mDigitizer.setGeometry(geom);

  mDigitizer.setDigits(mDigitsArrayPtr);
  mDigitizer.setMCLabels(mMCTruthArrayPtr);

  mDigitizer.init();

  return kSUCCESS;
}

//________________________________________________________
void DigitizerTask::Exec(Option_t* option)
{
  FairRootManager* mgr = FairRootManager::Instance();

  mDigitsArray.clear();
  mMCTruthArray.clear();

  double tEvent = mgr->GetEventTime() * mFairTimeUnitInNS; // event time in ns

  // is there QED backgroung provided? Fill QED slots until provided collision time
  if (mQEDBranch) {
    processQEDBackground(tEvent);
  }
  //
  mDigitizer.setEventTime(tEvent);
  // the type of digitization is steered by the DigiParams object of the Digitizer
  LOG(DEBUG) << "Running digitization on new event " << mEventID << " from source " << mSourceID;

  mDigitizer.process(mHitsArray, mEventID, mSourceID);

  mEventID++;
}

//________________________________________________________
void DigitizerTask::FinishTask()
{
  // finalize digitization, if needed, flash remaining digits

  if (!mDigitizer.getParams().isContinuous()) {
    return;
  }

  // is there QED backgroung provided? Fill QED slots up to the end of reserved ROFrames
  if (mQEDBranch) {
    processQEDBackground(mDigitizer.getEndTimeOfROFMax());
  }

  FairRootManager* mgr = FairRootManager::Instance();
  mgr->SetLastFill(kTRUE); /// necessary, otherwise the data is not written out
  mDigitsArray.clear();
  mMCTruthArray.clear();
  mDigitizer.fillOutputContainer();
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
    mDigitizer.process(mHitsArrayQED, mLastQEDEntry, mQEDSourceID);
    //
  }
}

//________________________________________________________
void DigitizerTask::setQEDInput(TBranch* qed, float timebin, UChar_t srcID)
{
  // assign the branch containing hits from QED electrons, whith every entry integrating
  // timebin ns of collisions

  LOG(INFO) << "Attaching QED ITS hits as sourceID=" << int(srcID) << ", entry integrates "
            << timebin << " ns";

  mQEDBranch = qed;
  mQEDEntryTimeBinNS = timebin;
  if (mQEDBranch) {
    assert(mQEDEntryTimeBinNS >= 1.0);
    mLastQEDTimeNS = -mQEDEntryTimeBinNS / 2; // time will be assigned to the middle of the bin
    mQEDBranch->SetAddress(&mHitsArrayQED);
    mLastQEDEntry = -1;
    mQEDSourceID = srcID;
    assert(mHitsArrayQED);
    assert(srcID < o2::MCCompLabel::maxSourceID());
  }
}
