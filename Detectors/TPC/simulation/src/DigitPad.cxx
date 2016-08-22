#include "TPCSimulation/DigitTime.h"
#include "TPCSimulation/DigitPad.h"
#include <iostream>
#include "TPCBase/Mapper.h"

#include "FairLogger.h"
using namespace AliceO2::TPC;

DigitPad::DigitPad(Int_t padID) :
mPadID(padID),
mTimeBins(500)
{}

DigitPad::~DigitPad(){
  for(std::vector<DigitTime*>::iterator iterTime = mTimeBins.begin(); iterTime != mTimeBins.end(); ++iterTime) {
    delete (*iterTime);
  }
}

void DigitPad::setDigit(Int_t time, Float_t charge){
  DigitTime *result = mTimeBins[time];
  if(result != nullptr){
    mTimeBins[time]->setDigit(charge);
  }
  else{
    mTimeBins[time] = new DigitTime(time);
//     digitTime = new DigitTime(time);
//     mTimeBins[time].push_back(digitTime);
    mTimeBins[time]->setDigit(charge);
  }
}

void DigitPad::reset(){
  for(std::vector<DigitTime*>::iterator iterTime = mTimeBins.begin(); iterTime != mTimeBins.end(); ++iterTime) {
    if((*iterTime) == nullptr) continue;
    (*iterTime)->reset();
  }
}

void DigitPad::fillOutputContainer(TClonesArray *output, Int_t cruID, Int_t rowID, Int_t padID){
  for(std::vector<DigitTime*>::iterator iterTime = mTimeBins.begin(); iterTime != mTimeBins.end(); ++iterTime) {
    if((*iterTime) == nullptr) continue;
    (*iterTime)->fillOutputContainer(output, cruID, rowID, padID, (*iterTime)->getTimeBin());
  }
}
