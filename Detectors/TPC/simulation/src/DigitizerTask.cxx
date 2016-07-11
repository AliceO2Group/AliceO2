//
//  DigitizerTask.cxx
//  ALICEO2
//
//  Created by Markus Fasel on 16.07.15.
//
//

#include "DigitizerTask.h"
#include "DigitContainer.h"  // for DigitContainer
#include "Digitizer.h"       // for Digitizer

#include "TObject.h"
#include "TClonesArray.h"
#include "FairLogger.h"
#include "FairRootManager.h"

ClassImp(AliceO2::TPC::DigitizerTask)

using namespace AliceO2::TPC;

DigitizerTask::DigitizerTask():
FairTask("TPCDigitizerTask"),
mDigitizer(nullptr),
mPointsArray(nullptr),
mDigitsArray(nullptr)
{
    mDigitizer = new Digitizer;
}

DigitizerTask::~DigitizerTask(){
    delete mDigitizer;
    if (mDigitsArray) delete mDigitsArray;
}

/// \brief Init function
/// Inititializes the digitizer and connects input and output container
InitStatus DigitizerTask::Init(){
    FairRootManager *mgr = FairRootManager::Instance();
    if(!mgr){
        LOG(ERROR) << "Could not instantiate FairRootManager. Exiting ..." << FairLogger::endl;
        return kERROR;
    }

    mPointsArray = dynamic_cast<TClonesArray *>(mgr->GetObject("TPCPoint"));
    if (!mPointsArray) {
        LOG(ERROR) << "TPC points not registered in the FairRootManager. Exiting ..." << FairLogger::endl;
        return kERROR;
    }

    // Register output container
    mDigitsArray = new TClonesArray("AliceO2::TPC::Digit");
    mgr->Register("TPCDigit", "TPC", mDigitsArray, kTRUE);      

    mDigitizer->init();
    return kSUCCESS;
}

void DigitizerTask::Exec(Option_t *option){
    mDigitsArray->Delete();
    LOG(DEBUG) << "Running digitization on new event" << FairLogger::endl;

    DigitContainer *digits = mDigitizer->Process(mPointsArray);
    digits->fillOutputContainer(mDigitsArray);
}
