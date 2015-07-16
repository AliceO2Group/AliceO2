/// \file AliITSUpgradeDigitizer.cxx
/// \brief Digitizer for the upgrated ITS
#include <iostream>

#include "TClonesArray.h"

#include "FairLink.h"
#include "FairLogger.h"

#include "Point.h"
#include "UpgradeGeometryTGeo.h"

#include "its/Digit.h"
#include "its/Digitizer.h"
#include "its/DigitContainer.h"

ClassImp(AliceO2::ITS::Digitizer)

using namespace AliceO2::ITS;

/// \brief Default constructor
///
/// Default constructor
Digitizer::Digitizer():
    TObject(),
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

void Digitizer::Init(){
    // Create the digit storage
    fDigitContainer = new DigitContainer(fGeometry);
}

    
/// \brief Exec function
///
/// Main function converting hits to digits
DigitContainer *Digitizer::Process(TClonesArray *points){
    fDigitContainer->Reset();
    
    // Convert points to summable digits
    for (TIter pointiter = TIter(points).Begin(); pointiter != TIter::End(); ++pointiter) {
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
    
    return fDigitContainer;
}
