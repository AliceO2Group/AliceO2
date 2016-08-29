#include "HitRow.h"
#include "HitPad.h"
#include "Mapper.h"

#include "FairLogger.h"
using namespace AliceO2::TPC;

HitRow::HitRow(Int_t rowID, Int_t npads):
mRowID(rowID),
mPads(npads)
{}

HitRow::~HitRow(){
  for(std::vector<HitPad*>::iterator iterPad = mPads.begin(); iterPad != mPads.end(); ++iterPad) {
    delete (*iterPad);
  }
}

void HitRow::setHit(Int_t pad, Int_t time, Float_t charge){
  HitPad *result = mPads[pad];
  if(result != nullptr){
    mPads[pad]->setHit(time, charge);
  }
  else{
    mPads[pad] = new HitPad(pad);
    mPads[pad]->setHit(time, charge);
  }
}

void HitRow::reset(){
  for(std::vector<HitPad*>::iterator iterPad = mPads.begin(); iterPad != mPads.end(); ++iterPad) {
    if((*iterPad) == nullptr) continue;
    (*iterPad)->reset();
  }
}

void HitRow::getHits(std::vector < AliceO2::TPC::PadHit* > &padHits, Int_t cruID, Int_t rowID){
  for(std::vector<HitPad*>::iterator iterPad = mPads.begin(); iterPad != mPads.end(); ++iterPad) {
    if((*iterPad) == nullptr) continue;
    (*iterPad)->getHits(padHits, cruID, rowID, (*iterPad)->getPad());
  }
}
