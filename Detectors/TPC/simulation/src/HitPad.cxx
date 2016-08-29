#include "TPCsimulation/HitTime.h"
#include "TPCsimulation/HitPad.h"
#include <iostream>
#include "TPCbase/Mapper.h"

#include "FairLogger.h"
using namespace AliceO2::TPC;

HitPad::HitPad(Int_t padID) :
mPadID(padID),
mTimeBins(500)
{}

HitPad::~HitPad(){
  for(std::vector<HitTime*>::iterator iterTime = mTimeBins.begin(); iterTime != mTimeBins.end(); ++iterTime) {
    delete (*iterTime);
  }
}

void HitPad::setHit(Int_t time, Float_t charge){
  HitTime *result = mTimeBins[time];
  if(result != nullptr){
    mTimeBins[time]->setHit(charge);
  }
  else{
    //if time bin outside specified range, the range of the vector is extended by one full drift time.
    while(int(mTimeBins.size()) <= time){
      mTimeBins.resize(int(mTimeBins.size()) + 500);
    }
    mTimeBins[time] = new HitTime(time);
    mTimeBins[time]->setHit(charge);
  }
}

void HitPad::reset(){
  for(std::vector<HitTime*>::iterator iterTime = mTimeBins.begin(); iterTime != mTimeBins.end(); ++iterTime) {
    if((*iterTime) == nullptr) continue;
    (*iterTime)->reset();
  }
}

void HitPad::getHits(std::vector < PadHit* > &padHits, Int_t cruID, Int_t rowID, Int_t padID){
  PadHit *hit = new PadHit(cruID, rowID, padID);
  
  for(std::vector<HitTime*>::iterator iterTime = mTimeBins.begin(); iterTime != mTimeBins.end(); ++iterTime) {
    if((*iterTime) == nullptr) continue;
    hit->addTimeHit((*iterTime)->getTimeBin(), (*iterTime)->getCharge(cruID, rowID, padID, (*iterTime)->getTimeBin()));
  }
  padHits.push_back(hit);
}
