#include "DigitTime.h"
#include "DigitADC.h"
#include "Digit.h"
#include "TClonesArray.h"
#include "Mapper.h"

#include "FairLogger.h"
using namespace AliceO2::TPC;

DigitTime::DigitTime(Int_t timeBin) :
mTimeBin(timeBin)
{}

DigitTime::~DigitTime(){
  for(std::vector<DigitADC*>::iterator iterADC = mADCCounts.begin(); iterADC != mADCCounts.end(); iterADC++) {
    delete (*iterADC);
  }   
}

void DigitTime::SetDigit(Float_t charge){
  digitAdc = new DigitADC(charge);
  mADCCounts.push_back(digitAdc);
}

void DigitTime::Reset(){
  mADCCounts.clear();
}

void DigitTime::FillOutputContainer(TClonesArray *output, Int_t cruID, Int_t rowID, Int_t padID, Int_t timeBin){
  mCharge = 0;
  for(std::vector<DigitADC*>::iterator iterADC = mADCCounts.begin(); iterADC != mADCCounts.end(); iterADC++) {
    if((*iterADC) == nullptr) continue;
    mCharge += (*iterADC)->GetADC();
  }
  Digit *digit = new Digit(cruID, mCharge, rowID, padID, timeBin);
  TClonesArray &clref = *output;
  new(clref[clref.GetEntriesFast()]) Digit(*(digit));
}