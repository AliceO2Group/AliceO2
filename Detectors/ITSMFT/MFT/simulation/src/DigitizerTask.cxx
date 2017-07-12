// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See https://alice-o2.web.cern.ch/ for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file DigitizerTask.h
/// \brief Task driving the convertion from Point to Digit
/// \author bogdan.vulpescu@cern.ch 
/// \date 03/05/2017

#include "FairLogger.h"

#include "MFTSimulation/DigitizerTask.h"

ClassImp(o2::MFT::DigitizerTask)

using namespace o2::MFT;

//_____________________________________________________________________________
DigitizerTask::DigitizerTask(Bool_t useAlpide)
  : FairTask("MFTDigitizerTask"), mUseAlpideSim(useAlpide), mDigitizer(), mPointsArray(nullptr), mDigitsArray(nullptr)
{

}

//_____________________________________________________________________________
DigitizerTask::~DigitizerTask()
{

  if (mDigitsArray) {
    mDigitsArray->Delete();
    delete mDigitsArray;
  }

}
/// \brief Init function
///
/// Inititializes the digitizer and connects input and output container
//_____________________________________________________________________________
InitStatus DigitizerTask::Init()
{

  FairRootManager* mgr = FairRootManager::Instance();
  if (!mgr) {
    LOG(ERROR) << "Could not instantiate FairRootManager. Exiting ..." << FairLogger::endl;
    return kERROR;
  }

  mPointsArray = dynamic_cast<TClonesArray*>(mgr->GetObject("MFTPoint"));
  if (!mPointsArray) {
    LOG(ERROR) << "MFT points not registered in the FairRootManager. Exiting ..." << FairLogger::endl;
    return kERROR;
  }

  // Register output container
  mDigitsArray = new TClonesArray("o2::ITSMFT::Digit");
  mgr->Register("MFTDigit", "MFT", mDigitsArray, kTRUE);

  mDigitizer.init(kTRUE);

  return kSUCCESS;

}

//_____________________________________________________________________________
void DigitizerTask::Exec(Option_t* option)
{

  mDigitsArray->Clear();
  LOG(DEBUG) << "Running digitization on new event" << FairLogger::endl;
  if (!mUseAlpideSim) {
    DigitContainer& digits = mDigitizer.process(mPointsArray);
    digits.fillOutputContainer(mDigitsArray);
  } else {
    mDigitizer.process(mPointsArray, mDigitsArray); // ALPIDE response
  }

}

