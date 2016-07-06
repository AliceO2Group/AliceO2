#include "DigitTime.h"
#include "DigitPad.h"
#include <iostream>
#include "Mapper.h"

#include "FairLogger.h"
using namespace AliceO2::TPC;

DigitPad::DigitPad(Int_t padID, Int_t nTimeBins) :
mPadID(padID),
mNTimeBins(nTimeBins),
mTimeBins(nTimeBins)
{}

DigitPad::~DigitPad(){
  for (int itime = 0; itime < mNTimeBins; itime++) {
    delete mTimeBins[itime];
  }
}

void DigitPad::SetDigit(Int_t time, Float_t charge){
  DigitTime *result = mTimeBins[time];
  if(result != nullptr){
    mTimeBins[time]->SetDigit(charge);
  }
  else{
    mTimeBins[time] = new DigitTime(time);
    mTimeBins[time]->SetDigit(charge);
  }
}

void DigitPad::Reset(){
  for(std::vector<DigitTime*>::iterator iterTime = mTimeBins.begin(); iterTime != mTimeBins.end(); iterTime++) {
    if((*iterTime) == nullptr) continue;
    (*iterTime)->Reset();
  }
}

void DigitPad::FillOutputContainer(TClonesArray *output, Int_t cruID, Int_t rowID, Int_t padID){
  for(std::vector<DigitTime*>::iterator iterTime = mTimeBins.begin(); iterTime != mTimeBins.end(); iterTime++) {
    if((*iterTime) == nullptr) continue;
    (*iterTime)->FillOutputContainer(output, cruID, rowID, padID, (*iterTime)->GetTimeBin());
  }
}
