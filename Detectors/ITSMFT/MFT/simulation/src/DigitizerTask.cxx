/// \file DigitizerTask.h
/// \brief Task driving the convertion from Hit to Digit
/// \author bogdan.vulpescu@cern.ch 
/// \date 03/05/2017

#include "FairLogger.h"

#include "ITSMFTSimulation/DigitContainer.h"

#include "MFTSimulation/DigitizerTask.h"

ClassImp(o2::MFT::DigitizerTask)

using o2::ITSMFT::DigitContainer;

using namespace o2::MFT;

//_____________________________________________________________________________
DigitizerTask::DigitizerTask(Bool_t useAlpide)
  : FairTask("MFTDigitizerTask"), mUseAlpideSim(useAlpide), mDigitizer(), mHitsArray(nullptr), mDigitsArray(nullptr)
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

  mHitsArray = dynamic_cast<TClonesArray*>(mgr->GetObject("MFTHits"));
  if (!mHitsArray) {
    LOG(ERROR) << "MFT hits not registered in the FairRootManager. Exiting ..." << FairLogger::endl;
    return kERROR;
  }

  // Register output container
  mDigitsArray = new TClonesArray("o2::ITSMFT::Digit");
  mgr->Register("MFTDigits", "MFT", mDigitsArray, kTRUE);

  mDigitizer.init(kTRUE);

  return kSUCCESS;

}

//_____________________________________________________________________________
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

