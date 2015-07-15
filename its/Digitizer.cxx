/// \file AliITSUpgradeDigitizer.cxx
/// \brief Digitizer for the upgrated ITS
#include <iostream>

#include "TClonesArray.h"

#include "FairLink.h"
#include "FairLogger.h"
#include "FairRootManager.h"

#include "Point.h"
#include "UpgradeGeometryTGeo.h"

#include "Digit.h"
#include "Digitizer.h"
#include "DigitContainer.h"

ClassImp(AliceO2::ITS::Digitizer)

using namespace AliceO2::ITS;

/// \brief Default constructor
///
/// Default constructor
Digitizer::Digitizer():
    FairTask("ITSDigitizer"),
    fPointsArray(nullptr),
    fDigitsArray(nullptr),
    fDigitContainer(nullptr),
    fGain(1.)
{
    fGeometry = new UpgradeGeometryTGeo(kTRUE, kTRUE);
}
    
/// \brief Destructor
///
/// Destructor
Digitizer::~Digitizer(){
    delete fGeometry;
    if(fDigitContainer) delete fDigitContainer;
}
    
/// \brief Init function
///
/// Inititializes the digitizer and connects input and output container
InitStatus Digitizer::Init(){
    FairRootManager *mgr = FairRootManager::Instance();
    if(!mgr){
        LOG(ERROR) << "Could not instantiate FairRootManager. Exiting ..." << FairLogger::endl;
        return kERROR;
    }
        
    fPointsArray = dynamic_cast<TClonesArray *>(mgr->GetObject("Point"));
    if (!fPointsArray) {
        LOG(ERROR) << "ITS points not registered in the FairRootManager. Exiting ..." << FairLogger::endl;
        return kERROR;
    }
        
    // Register output container
    fDigitsArray = new TClonesArray("AliceO2::ITS::Digit");
    mgr->Register("Digit", "ITS", fDigitsArray, kTRUE);
    
    // Create the digit storage
    fDigitContainer = new DigitContainer(fGeometry);
    return kSUCCESS;
}
    
/// \brief Exec function
///
/// Main function converting hits to digits
/// \param option Optional further settings
void Digitizer::Exec(Option_t *option){
    fDigitsArray->Delete();
    LOG(DEBUG) << "Running digitization on new event" << FairLogger::endl;
    
    // Convert points to summable digits
    for (TIter pointiter = TIter(fPointsArray).Begin(); pointiter != TIter::End(); ++pointiter) {
        Point *inputpoint = dynamic_cast<Point *>(*pointiter);
        LOG(DEBUG) << "Processing next point: " << FairLogger::endl;
        LOG(DEBUG) << "=======================" << FairLogger::endl;
        LOG(DEBUG) << *inputpoint << FairLogger::endl;
        Int_t layer = fGeometry->getLayer(inputpoint->GetDetectorID()),
                stave = fGeometry->getStave(inputpoint->GetDetectorID()),
                staveindex = fGeometry->getChipIdInStave(inputpoint->GetDetectorID());
        Double_t charge = inputpoint->GetEnergyLoss();
        Digit * digit = fDigitContainer->FindDigit(layer, stave, staveindex);
        if(digit){
            // For the moment only add the charge to the current digit
            LOG(DEBUG) << "Digit already found" << FairLogger::endl;
            double chargeDigit = digit->GetCharge();
            chargeDigit += charge * fGain;
        } else {
            LOG(DEBUG) << "Creating new digit" << FairLogger::endl;
            digit = new Digit(inputpoint->GetDetectorID(), charge, inputpoint->GetTime());
            fDigitContainer->AddDigit(digit);
        }
    }
    
    // Fill output container
    fDigitContainer->FillOutputContainer(fDigitsArray);
    fDigitContainer->Reset();
}
