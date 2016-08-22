#include "HitContainer.h"
#include "HitCRU.h"
#include "Mapper.h"
#include "CRU.h"
#include "PadHit.h"

#include "FairLogger.h"

#include "TClonesArray.h"

using namespace AliceO2::TPC;

HitContainer::HitContainer():
mCRU(CRU::MaxCRU)
{}

HitContainer::~HitContainer(){
  for(std::vector<HitCRU*>::iterator iterCRU = mCRU.begin(); iterCRU != mCRU.end(); ++iterCRU) {
    delete (*iterCRU);
  }
}

void HitContainer::reset(){
  for(std::vector<HitCRU*>::iterator iterCRU = mCRU.begin(); iterCRU != mCRU.end(); ++iterCRU) {
    if((*iterCRU) == nullptr) continue;
    (*iterCRU)->reset();
  }
}

void HitContainer::addHit(Int_t cru, Int_t row, Int_t pad, Int_t time, Float_t charge){
  HitCRU *result = mCRU[cru];
  if(result != nullptr){
    mCRU[cru]->setHit(row, pad, time, charge);
  }
  else{
    const Mapper& mapper = Mapper::instance();
    mCRU[cru] = new HitCRU(cru, mapper.getPadRegionInfo(CRU(cru).region()).getNumberOfPadRows());
    mCRU[cru]->setHit(row, pad, time, charge);
  }
}


void HitContainer::getHits(std::vector < AliceO2::TPC::PadHit* > &padHits){
  for(std::vector<HitCRU*>::iterator iterCRU = mCRU.begin(); iterCRU != mCRU.end(); ++iterCRU) {
    if((*iterCRU) == nullptr) continue;
    (*iterCRU)->getHits(padHits, (*iterCRU)->getCRUID());
  }
}
