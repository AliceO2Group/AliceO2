#include "TPCsimulation/HitTime.h"
#include "TPCsimulation/HitCharge.h"
#include "TClonesArray.h"

#include "FairLogger.h"
using namespace AliceO2::TPC;

HitTime::HitTime(Int_t timeBin) :
mTimeBin(timeBin),
mChargeCounts()
{}

HitTime::~HitTime(){
  for(std::vector<HitCharge*>::iterator iterCharge = mChargeCounts.begin(); iterCharge != mChargeCounts.end(); ++iterCharge) {
    delete (*iterCharge);
  }   
}

void HitTime::setHit(Float_t charge){
  hitCharge = new HitCharge(charge);
  mChargeCounts.push_back(hitCharge);
}

void HitTime::reset(){
  mChargeCounts.clear();
}

Double_t HitTime::getCharge(Int_t cruID, Int_t rowID, Int_t padID, Int_t timeBin){
  Double_t charge = 0;
  for(std::vector<HitCharge*>::iterator iterCharge = mChargeCounts.begin(); iterCharge != mChargeCounts.end(); ++iterCharge) {
    if((*iterCharge) == nullptr) continue;
    charge += (*iterCharge)->getCharge();
  }
  return charge;
}