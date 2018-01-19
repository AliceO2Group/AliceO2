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


#include "TOFSimulation/DigitizerTask.h"
#include "DetectorsBase/Utils.h"

#include "FairLogger.h"      // for LOG
#include "FairRootManager.h" // for FairRootManager


ClassImp(o2::tof::DigitizerTask);

using namespace o2::tof;

DigitizerTask::DigitizerTask() : FairTask("TOFDigitizerTask"), mDigitizer() {}
DigitizerTask::~DigitizerTask()
{
  if (mDigitsArray) {
    mDigitsArray->clear();
    delete mDigitsArray;
  }
}

/// Inititializes the digitizer and connects input and output container
InitStatus DigitizerTask::Init()
{
  FairRootManager* mgr = FairRootManager::Instance();
  if (!mgr) {
    LOG(ERROR) << "Could not instantiate FairRootManager. Exiting ..." << FairLogger::endl;
    return kERROR;
  }

  mHitsArray = mgr->InitObjectAs<const std::vector<o2::tof::HitType>*>("TOFHit");
  if (!mHitsArray) {
    LOG(ERROR) << "TOF hits not registered in the FairRootManager. Exiting ..." << FairLogger::endl;
    return kERROR;
  }

  // Register output container
  mgr->RegisterAny("TOFDigit", mDigitsArray, kTRUE);

  // Register MC Truth container
  mMCTruthArray = new typename std::remove_pointer<decltype(mMCTruthArray)>::type;
  mgr->RegisterAny("TOFDigitMCTruth", mMCTruthArray, kTRUE);

//  mDigitizer.setCoeffToNanoSecond(mFairTimeUnitInNS);

//  mDigitizer.init();
  return kSUCCESS;
}

//________________________________________________________
void DigitizerTask::Exec(Option_t* option)
{
  FairRootManager* mgr = FairRootManager::Instance();

  if (mDigitsArray) mDigitsArray->clear();
  mDigitizer.setEventTime(mgr->GetEventTime());

  if (mMCTruthArray) {
    mMCTruthArray->clear();
  }

  o2::dataformats::MCTruthContainer<o2::tof::MCLabel> transientTruthContainer;
  mDigitizer.setMCTruthContainer(&transientTruthContainer);

  // the type of digitization is steered by the DigiParams object of the Digitizer
  LOG(DEBUG) << "Running digitization on new event " << mEventID
             << " from source " << mSourceID << FairLogger::endl;

  /// RS: ATTENTION: this is just a trick until we clarify how the hits from different source are
  /// provided and identified.
  mDigitizer.setSrcID(mSourceID);
  mDigitizer.setEventID(mEventID);

  LOG(INFO) << "Digitizing " << mHitsArray->size() << " hits \n";
  mDigitizer.process(mHitsArray, mDigitsArray);

  // copying the transient labels to the output labels (stripping the tdc information)
  if (mMCTruthArray) {
    // copy from transientTruthContainer to mMCTruthAray
    // a brute force solution for the moment; should be handled by a dedicated API
    for (int index = 0; index < transientTruthContainer.getIndexedSize(); ++index) {
      auto labels = transientTruthContainer.getLabels(index);
      for (auto& l : labels) {
        mMCTruthArray->addElement(index, l);
      }
    }
  }

  mEventID++;
}

//________________________________________________________
void DigitizerTask::FinishTask()
{
  // finalize digitization, if needed, flash remaining digits
  if(!mContinuous) return;
  FairRootManager *mgr = FairRootManager::Instance();
  mgr->SetLastFill(kTRUE); /// necessary, otherwise the data is not written out
  if (mDigitsArray) mDigitsArray->clear();

  // TODO: reenable this
  //mDigitizer.fillOutputContainer(mDigitsArray);
}




