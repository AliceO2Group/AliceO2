#include "TPCSimulation/DigitTime.h"
#include "TPCSimulation/DigitADC.h"
#include "TPCSimulation/Digit.h"
#include "TClonesArray.h"
#include "FairLogger.h"
using namespace AliceO2::TPC;

DigitTime::DigitTime(Int_t timeBin) :
mTimeBin(timeBin)
{}

DigitTime::~DigitTime(){
  for(std::vector<DigitADC*>::iterator iterADC = mADCCounts.begin(); iterADC != mADCCounts.end(); ++iterADC) {
    delete (*iterADC);
  }
}

void DigitTime::setDigit(Float_t charge){
  digitAdc = new DigitADC(charge);
  mADCCounts.push_back(digitAdc);
}

void DigitTime::reset(){
  mADCCounts.clear();
}

void DigitTime::fillOutputContainer(TClonesArray *output, Int_t cruID, Int_t rowID, Int_t padID, Int_t timeBin){
  //TODO: Store parameters elsewhere
  Float_t ADCSat = 1023;
 
  mADC = 0;
  for(std::vector<DigitADC*>::iterator iterADC = mADCCounts.begin(); iterADC != mADCCounts.end(); ++iterADC) {
    if((*iterADC) == nullptr) continue;
    mADC += (*iterADC)->getADC();
  }

  if(mADC > 0){
    if(mADC >= ADCSat) mADC = ADCSat-1;// saturation
    Digit *digit = new Digit(cruID, mADC, rowID, padID, timeBin);
    TClonesArray &clref = *output;
    new(clref[clref.GetEntriesFast()]) Digit(*(digit));
  }
}
