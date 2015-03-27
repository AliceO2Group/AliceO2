//
//  DigitStave.cxx
//  ALICEO2
//
//  Created by Markus Fasel on 26.03.15.
//
//

#include "FairLogger.h"
#include "Digit.h"
#include "DigitStave.h"

using namespace AliceO2::ITS;

DigitStave::DigitStave(Int_t npixel){
    
}

DigitStave::~DigitStave() {}

void DigitStave::Reset(){
    fPixels.clear();
}

Digit *DigitStave::FindDigit(Int_t pixel){
    Digit *result = nullptr;
    auto digitentry = fPixels.find(pixel);
    if (digitentry != fPixels.end()) {
        result = digitentry->second;
    }
    return result;
}

void DigitStave::SetDigit(int pixel, Digit *digi){
    Digit *olddigit = FindDigit(pixel);
    if(olddigit != nullptr){
        LOG(ERROR) << "Pixel already set previously, replacing and deleting previous content" << FairLogger::endl;
        delete olddigit;
    }
    fPixels.insert(std::pair<int, Digit *>(pixel, digi));
}