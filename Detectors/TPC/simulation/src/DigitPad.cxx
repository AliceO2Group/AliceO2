#include "TPCSimulation/DigitPad.h"
#include "TPCSimulation/Digitizer.h"
#include "TPCSimulation/Digit.h"

#include "TClonesArray.h"
#include "FairLogger.h"
using namespace AliceO2::TPC;

DigitPad::DigitPad(Int_t pad) :
mPad(pad)
{}

DigitPad::~DigitPad() {
  mADCCounts.resize(0);
}

void DigitPad::fillOutputContainer(TClonesArray *output, Int_t cru, Int_t timeBin, Int_t row, Int_t pad) {  
  Float_t mCharge = 0;
  for(auto &aADCCounts : mADCCounts) {
    mCharge += aADCCounts.getADC();
  }
  
  Digitizer d;
  const Int_t mADC = d.ADCvalue(mCharge);
  
  if(mADC > 0) {
    auto *digit = new Digit(cru, mADC, row, pad, timeBin);
    TClonesArray &clref = *output;
    new(clref[clref.GetEntriesFast()]) Digit(*(digit));
  }
}
