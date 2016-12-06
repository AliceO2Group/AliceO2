//
//  DigitizerTask.cxx
//  ALICEO2
//
//  Created by Markus Fasel on 16.07.15.
//
//

#include "ITSSimulation/DigitizerTask.h"
#include "ITSSimulation/DigitContainer.h"
#include "ITSSimulation/Digitizer.h"

#include "TObject.h"             // for TObject
#include "TClonesArray.h"        // for TClonesArray
#include "FairLogger.h"          // for LOG
#include "FairRootManager.h"     // for FairRootManager

ClassImp(AliceO2::ITS::DigitizerTask)

using namespace AliceO2::ITS;

DigitizerTask::DigitizerTask(Bool_t useAlpide) :
  FairTask("ITSDigitizerTask"),
  fUseAlpideSim(useAlpide),
  fDigitizer(nullptr),
  fPointsArray(nullptr),
  fDigitsArray(nullptr)
{
}

DigitizerTask::~DigitizerTask()
{
  delete fDigitizer;
  if (fDigitsArray) { delete fDigitsArray; }
}

/// \brief Init function
///
/// Inititializes the digitizer and connects input and output container
InitStatus DigitizerTask::Init()
{
  FairRootManager *mgr = FairRootManager::Instance();
  if (!mgr) {
    LOG(ERROR) << "Could not instantiate FairRootManager. Exiting ..." << FairLogger::endl;
    return kERROR;
  }

  fPointsArray = dynamic_cast<TClonesArray *>(mgr->GetObject("ITSPoint"));
  if (!fPointsArray) {
    LOG(ERROR) << "ITS points not registered in the FairRootManager. Exiting ..." << FairLogger::endl;
    return kERROR;
  }

  // Register output container
  fDigitsArray = new TClonesArray("AliceO2::ITS::Digit");
  mgr->Register("ITSDigit", "ITS", fDigitsArray, kTRUE);
  
  fDigitizer = new Digitizer;

  return kSUCCESS;
}

void DigitizerTask::Exec(Option_t *option)
{
  fDigitsArray->Clear();
  LOG(DEBUG) << "Running digitization on new event" << FairLogger::endl;
  if (!fUseAlpideSim) {
     DigitContainer *digits = fDigitizer->Process(fPointsArray);
     digits->FillOutputContainer(fDigitsArray);
  } else {
     fDigitizer->Process(fPointsArray,fDigitsArray); // ALPIDE response
  }
}
