#include "TPCSimulation/DigitTime.h"
#include "TPCSimulation/DigitADC.h"
#include "TPCSimulation/Digit.h"
#include "TClonesArray.h"
#include "TPCBase/Mapper.h"

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
  mCharge = 0;
  for(std::vector<DigitADC*>::iterator iterADC = mADCCounts.begin(); iterADC != mADCCounts.end(); ++iterADC) {
    if((*iterADC) == nullptr) continue;
    mCharge += (*iterADC)->getADC();
  }
  //TODO have to understand what is going wrong here - tree is filled with many zeros otherwise...
  if(mCharge > 0){
    if(mCharge > 1024) mCharge = 1024;
//     Digit *digit = new Digit(cruID, mCharge, rowID, padID, timeBin);
    TClonesArray &clref = *output;
//     new(clref[clref.GetEntriesFast()]) Digit(*(digit));
    new(clref[clref.GetEntriesFast()]) Digit(cruID, mCharge, rowID, padID, timeBin);
  }
}
