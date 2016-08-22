#include "PadHit.h"
#include "PadHitTime.h"

ClassImp(AliceO2::TPC::PadHit)

using namespace AliceO2::TPC;


PadHit::PadHit():
mCRU(),
mRow(),
mPad(), 
mTimeHits(0)
{
}

PadHit::PadHit(Int_t cru, Int_t row, Int_t pad):
mCRU(cru),
mRow(row),
mPad(pad), 
mTimeHits(0)
{
}

PadHit::~PadHit(){  
  for(std::vector<PadHitTime*>::iterator iter = mTimeHits.begin(); iter != mTimeHits.end(); ++iter) {
    delete (*iter);
  }
}

void PadHit::addTimeHit(Double_t time, Double_t charge){
  PadHitTime *hitTime = new PadHitTime(time, charge);
  mTimeHits.push_back(hitTime);
}