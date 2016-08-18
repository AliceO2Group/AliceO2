#include "TPCsimulation/DigitCRU.h"
#include "TPCsimulation/DigitRow.h"
#include "TPCsimulation/DigitPad.h"
#include "TPCbase/Mapper.h"

#include "FairLogger.h"
using namespace AliceO2::TPC;

#include <iostream>

DigitCRU::DigitCRU(Int_t cruID, Int_t nrows):
    mCRUID(cruID),
    mNRows(nrows),
    mRows(nrows)
{}

DigitCRU::~DigitCRU(){
  for (int irow = 0; irow < mNRows; ++irow) {
    delete mRows[irow];
  }
}

void DigitCRU::setDigit(Int_t row, Int_t pad, Int_t time, Float_t charge){
  DigitRow *result = mRows[row];
  if(result != nullptr){
    mRows[row]->setDigit(pad, time, charge);
  }
  else{
    const Mapper& mapper = Mapper::instance();
    mRows[row] = new DigitRow(row, mapper.getPadRegionInfo(CRU(mCRUID).region()).getPadsInRowRegion(row));
    mRows[row]->setDigit(pad, time, charge);
  }
}

void DigitCRU::reset(){
    for(std::vector<DigitRow*>::iterator iterRow = mRows.begin(); iterRow != mRows.end(); ++iterRow) {
      if((*iterRow) == nullptr) continue;
      (*iterRow)->reset();
    }
}

void DigitCRU::fillOutputContainer(TClonesArray *output, Int_t cruID){
    for(std::vector<DigitRow*>::iterator iterRow = mRows.begin(); iterRow != mRows.end(); ++iterRow) {
      if((*iterRow) == nullptr) continue;
      (*iterRow)->fillOutputContainer(output, cruID, (*iterRow)->getRow());
    }
}
