#include "TPCSimulation/DigitTime.h"
#include "TPCSimulation/DigitPad.h"
#include "FairLogger.h"
using namespace AliceO2::TPC;

DigitPad::DigitPad(Int_t padID) :
mPadID(padID),
mTimeBins(500)
{}

DigitPad::~DigitPad()
{
  for(auto iterTime = mTimeBins.begin(); iterTime != mTimeBins.end(); ++iterTime) {
    delete (*iterTime);
  }
}

void DigitPad::setDigit(Int_t time, Float_t charge)
{
  //if time bin outside specified range, the range of the vector is extended by one full drift time.
  while(static_cast<int>(mTimeBins.size()) <= time){
    mTimeBins.resize(static_cast<int>(mTimeBins.size()) + 500);
  }
  
  DigitTime *result = mTimeBins[time];
  if(result != nullptr){
    mTimeBins[time]->setDigit(charge);
  }
  else{
    mTimeBins[time] = new DigitTime(time);
    mTimeBins[time]->setDigit(charge);
  }
}

void DigitPad::reset()
{
  for(auto iterTime = mTimeBins.begin(); iterTime != mTimeBins.end(); ++iterTime) {
    if((*iterTime) == nullptr) continue;
    (*iterTime)->reset();
  }
}

void DigitPad::fillOutputContainer(TClonesArray *output, Int_t cruID, Int_t rowID, Int_t padID)
{
  for(auto iterTime = mTimeBins.begin(); iterTime != mTimeBins.end(); ++iterTime) {
    if((*iterTime) == nullptr) continue;
    (*iterTime)->fillOutputContainer(output, cruID, rowID, padID, (*iterTime)->getTimeBin());
  }
}
