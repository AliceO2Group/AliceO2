/// \file DigitizerTask.cxx
/// \brief Implementation of the ITS digitizer task

//
//  Created by Markus Fasel on 16.07.15.
//
//

#include "ITSSimulation/DigitizerTask.h"
#include "ITSSimulation/DigitContainer.h"

#include "FairLogger.h"      // for LOG
#include "FairRootManager.h" // for FairRootManager
#include "TClonesArray.h"    // for TClonesArray
#include "TObject.h"         // for TObject

ClassImp(o2::ITS::DigitizerTask)

  using namespace o2::ITS;

DigitizerTask::DigitizerTask(Bool_t useAlpide)
  : FairTask("ITSDigitizerTask"), mUseAlpideSim(useAlpide), mDigitizer(), mPointsArray(nullptr), mDigitsArray(nullptr)
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

  mPointsArray = dynamic_cast<TClonesArray*>(mgr->GetObject("ITSPoint"));
  if (!mPointsArray) {
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
    DigitContainer& digits = mDigitizer.process(mPointsArray);
    digits.fillOutputContainer(mDigitsArray);
  } else {
    mDigitizer.process(mPointsArray, mDigitsArray); // ALPIDE response
  }
}
