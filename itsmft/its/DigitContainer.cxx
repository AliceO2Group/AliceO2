//
//  DigitContainer.cxx
//  ALICEO2
//
//  Created by Markus Fasel on 25.03.15.
//
//
#include "TClonesArray.h"
#include "FairLogger.h"

#include "itsmft/its/Digit.h"
#include "itsmft/its/DigitLayer.h"
#include "itsmft/its/DigitContainer.h"


using namespace AliceO2::ITS;


DigitContainer::DigitContainer(const UpgradeGeometryTGeo *geo):
fGeometry(geo)
{
     for (Int_t ily =0; ily < 7; ily++) {
        fDigitLayer[ily] = new DigitLayer(ily,
                                          fGeometry->getNumberOfStaves(ily));
    }
}

DigitContainer::~DigitContainer(){
    for(int ily = 0; ily < 7; ily++) delete fDigitLayer[ily];
}

void DigitContainer::Reset(){
    for(int ily = 0; ily < 7; ily++) fDigitLayer[ily]->Reset();
}

Digit *DigitContainer::FindDigit(int layer, int stave, int pixel){
    if (layer >= 7) {
        LOG(ERROR) << "Layer index out of range : " << layer << ", max 6" << FairLogger::endl;
        return nullptr;
    }
    return fDigitLayer[layer]->FindDigit(stave, pixel);
}

void DigitContainer::AddDigit(Digit *digi){
    Int_t layer = fGeometry->getLayer(digi->GetChipIndex()),
    stave = fGeometry->getStave(digi->GetChipIndex()),
    pixel = fGeometry->getChipIdInStave(digi->GetChipIndex());
    
    if(layer >= 7){
        LOG(ERROR) << "Layer index out of range : " << layer << ", max 6" << FairLogger::endl;
        return;
    }
    fDigitLayer[layer]->SetDigit(digi, stave, pixel);
}

void DigitContainer::FillOutputContainer(TClonesArray *output){
    for(DigitLayer **layeriter = fDigitLayer; layeriter < fDigitLayer + 7; layeriter++){
        (*layeriter)->FillOutputContainer(output);
    }
}
