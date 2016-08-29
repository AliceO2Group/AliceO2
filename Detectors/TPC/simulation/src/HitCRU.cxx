#include "TPCSimulation/HitCRU.h"
#include "TPCSimulation/HitRow.h"
#include "TPCBase/Mapper.h"

#include "FairLogger.h"
using namespace AliceO2::TPC;

#include <iostream>

HitCRU::HitCRU(Int_t cruID, Int_t nrows):
mCRUID(cruID),
mNRows(nrows),
mRows(nrows)
{}

HitCRU::~HitCRU(){
  for(std::vector<HitRow*>::iterator iterRow = mRows.begin(); iterRow != mRows.end(); ++iterRow) {
    delete (*iterRow);
  }
}

void HitCRU::setHit(Int_t row, Int_t pad, Int_t time, Float_t charge){
  HitRow *result = mRows[row];
  if(result != nullptr){
    mRows[row]->setHit(pad, time, charge);
  }
  else{
    const Mapper& mapper = Mapper::instance();
    mRows[row] = new HitRow(row, mapper.getPadRegionInfo(CRU(mCRUID).region()).getPadsInRowRegion(row));
    mRows[row]->setHit(pad, time, charge);
  }
}

void HitCRU::reset(){
  for(std::vector<HitRow*>::iterator iterRow = mRows.begin(); iterRow != mRows.end(); ++iterRow) {
    if((*iterRow) == nullptr) continue;
    (*iterRow)->reset();
  }
}

void HitCRU::getHits(std::vector < AliceO2::TPC::PadHit* > &padHits, Int_t cruID){
  for(std::vector<HitRow*>::iterator iterRow = mRows.begin(); iterRow != mRows.end(); ++iterRow) {
    if((*iterRow) == nullptr) continue;
    (*iterRow)->getHits(padHits, cruID, (*iterRow)->getRow());
  }
}
