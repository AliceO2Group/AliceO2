// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See https://alice-o2.web.cern.ch/ for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file DigitizerTask.cxx
/// \brief Implementation of the ITS digitizer task

//
//  Created by Markus Fasel on 16.07.15.
//
//

#include "ITSMFTSimulation/DigitContainer.h"
#include "ITSSimulation/DigitizerTask.h"

#include "FairLogger.h"      // for LOG
#include "FairRootManager.h" // for FairRootManager
#include "TClonesArray.h"    // for TClonesArray
#include "TObject.h"         // for TObject

ClassImp(o2::ITS::DigitizerTask)

using o2::ITSMFT::DigitContainer;
using namespace o2::ITS;

DigitizerTask::DigitizerTask(Bool_t useAlpide)
  : FairTask("ITSDigitizerTask"), mUseAlpideSim(useAlpide), mDigitizer(), mHitsArray(nullptr), mDigitsArray(nullptr)
{
}

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
InitStatus DigitizerTask::Init()
{
  FairRootManager* mgr = FairRootManager::Instance();
  if (!mgr) {
    LOG(ERROR) << "Could not instantiate FairRootManager. Exiting ..." << FairLogger::endl;
    return kERROR;
  }

  mHitsArray = dynamic_cast<TClonesArray*>(mgr->GetObject("ITSHit"));
  if (!mHitsArray) {
    LOG(ERROR) << "ITS points not registered in the FairRootManager. Exiting ..." << FairLogger::endl;
    return kERROR;
  }

  // Register output container
  mDigitsArray = new TClonesArray("o2::ITSMFT::Digit");
  mgr->Register("ITSDigit", "ITS", mDigitsArray, kTRUE);

  mDigitizer.init(kTRUE);

  return kSUCCESS;
}

void DigitizerTask::Exec(Option_t* option)
{
  mDigitsArray->Clear();
  LOG(DEBUG) << "Running digitization on new event" << FairLogger::endl;
  if (!mUseAlpideSim) {
    DigitContainer& digits = mDigitizer.process(mHitsArray);
    digits.fillOutputContainer(mDigitsArray);
  } else {
    mDigitizer.process(mHitsArray, mDigitsArray); // ALPIDE response
  }
}
