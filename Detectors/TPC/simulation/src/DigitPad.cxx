#include "TPCSimulation/DigitPad.h"
#include "TPCSimulation/Digitizer.h"
#include "TPCSimulation/Digit.h"
#include <iostream>

#include "TClonesArray.h"
#include "FairLogger.h"
using namespace AliceO2::TPC;

DigitPad::DigitPad(Int_t pad) :
mPad(pad),
mTotalChargePad(0.)
{}

DigitPad::~DigitPad() {
  mADCCounts.resize(0);
  mTotalChargePad = 0;
}

void DigitPad::fillOutputContainer(TClonesArray *output, Int_t cru, Int_t timeBin, Int_t row, Int_t pad) {  
  for(auto &aADCCounts : mADCCounts) {
    mTotalChargePad += aADCCounts.getADC();
  }
  
  Digitizer d;
  const Float_t mADC = d.ADCvalue(mTotalChargePad);
  
  if(mADC > 0) {
    auto *digit = new Digit(cru, mADC, row, pad, timeBin);
    TClonesArray &clref = *output;
    new(clref[clref.GetEntriesFast()]) Digit(*(digit));
  }
}

void DigitPad::fillOutputContainer(TClonesArray *output, Int_t cru, Int_t timeBin, Int_t row, Int_t pad, std::vector<CommonMode> commonModeContainer) {  
  for(auto &aADCCounts : mADCCounts) {
    mTotalChargePad += aADCCounts.getADC();
  }
  
  Digitizer d;
  const Float_t mADC = d.ADCvalue(mTotalChargePad);
  
  if(mADC > 0) {
    Float_t commonMode =0;
    for (auto &aCommonMode :commonModeContainer){
      if(aCommonMode.getCRU() == cru && aCommonMode.getTimeBin() == timeBin) commonMode = aCommonMode.getCommonMode();
    }
    Digit *digit = new Digit(cru, mADC, row, pad, timeBin, commonMode);
    TClonesArray &clref = *output;
    new(clref[clref.GetEntriesFast()]) Digit(*(digit));
  }
}

void DigitPad::processCommonMode(Int_t cru, Int_t timeBin, Int_t row, Int_t pad) {  
  for(auto &aADCCounts : mADCCounts) {
    mTotalChargePad += aADCCounts.getADC();
  }
}
