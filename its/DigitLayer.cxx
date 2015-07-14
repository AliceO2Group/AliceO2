//
//  DigitLayer.cxx
//  ALICEO2
//
//  Created by Markus Fasel on 25.03.15.
//
//

#include "FairLogger.h"

#include "DigitLayer.h"
#include "DigitStave.h"

using namespace AliceO2::ITS;

DigitLayer::DigitLayer(Int_t layerID, Int_t nstaves, Int_t npixels):
    fLayerID(layerID),
    fNStaves(nstaves),
    fStaves(nullptr)
{
    fStaves = new DigitStave*[fNStaves];
    for (int istave = 0; istave < fNStaves; istave++) {
        fStaves[istave] = new DigitStave(npixels);
    }
}

DigitLayer::~DigitLayer(){
    for (int istave = 0; istave < 0; istave++) {
        delete fStaves[istave];
    }
}

void DigitLayer::SetDigit(Digit* digi, Int_t stave, Int_t pixel){
    if (stave >= fNStaves) {
        LOG(ERROR) << "Stave index " << stave << " out of range for layer " << fLayerID << ", maximum " << fNStaves << FairLogger::endl;
    } else {
        fStaves[stave]->SetDigit(pixel, digi);
    }
}

Digit *DigitLayer::FindDigit(Int_t stave, Int_t pixel){
    if (stave > fNStaves) {
        LOG(ERROR) << "Stave index " << stave << " out of range for layer " << fLayerID << ", maximum " << fNStaves << FairLogger::endl;
        return nullptr;
    }
    return fStaves[stave]->FindDigit(pixel);
}