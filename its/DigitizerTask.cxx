//
//  DigitizerTask.cxx
//  ALICEO2
//
//  Created by Markus Fasel on 16.07.15.
//
//

#include "its/DigitizerTask.h"
#include <TClonesArray.h>        // for TClonesArray
#include "FairLogger.h"          // for LOG
#include "FairRootManager.h"     // for FairRootManager
#include "TObject.h"             // for TObject
#include "its/DigitContainer.h"  // for DigitContainer
#include "its/Digitizer.h"       // for Digitizer

ClassImp(AliceO2::ITS::DigitizerTask)

using namespace AliceO2::ITS;

DigitizerTask::DigitizerTask():
FairTask("ITSDigitizerTask"),
fDigitizer(nullptr),
fPointsArray(nullptr),
fDigitsArray(nullptr)
{
    fDigitizer = new Digitizer;
}

DigitizerTask::~DigitizerTask(){
    delete fDigitizer;
    if (fDigitsArray) delete fDigitsArray;
}

/// \brief Init function
///
/// Inititializes the digitizer and connects input and output container
InitStatus DigitizerTask::Init(){
    FairRootManager *mgr = FairRootManager::Instance();
    if(!mgr){
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
    
    fDigitizer->Init();
    return kSUCCESS;
}

void DigitizerTask::Exec(Option_t *option){
    fDigitsArray->Delete();
    LOG(DEBUG) << "Running digitization on new event" << FairLogger::endl;
    
    DigitContainer *digits = fDigitizer->Process(fPointsArray);
    digits->FillOutputContainer(fDigitsArray);
}

