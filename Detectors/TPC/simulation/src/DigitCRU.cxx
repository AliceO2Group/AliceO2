#include "DigitCRU.h"
#include "DigitRow.h"
#include "DigitPad.h"
#include "Mapper.h"

#include "FairLogger.h"
using namespace AliceO2::TPC;

#include <iostream>

DigitCRU::DigitCRU(Int_t cruID, Int_t nrows):
    mCRUID(cruID),
    mNRows(nrows),
    mRows(nrows)
{}

DigitCRU::~DigitCRU(){
  for (int irow = 0; irow < mNRows; irow++) {
    delete mRows[irow];
  }
}

void DigitCRU::SetDigit(Int_t row, Int_t pad, Int_t time, Float_t charge){
  DigitRow *result = mRows[row];
  if(result != nullptr){
    mRows[row]->SetDigit(pad, time, charge);
  }
  else{
    const Mapper& mapper = Mapper::instance();
    mRows[row] = new DigitRow(row, mapper.getPadRegionInfo(CRU(mCRUID).region()).getPadsInRowRegion(row));
    mRows[row]->SetDigit(pad, time, charge);
  }
}

void DigitCRU::Reset(){
    for(std::vector<DigitRow*>::iterator iterRow = mRows.begin(); iterRow != mRows.end(); iterRow++) {
      if((*iterRow) == nullptr) continue;
      (*iterRow)->Reset();
    }
}

void DigitCRU::FillOutputContainer(TClonesArray *output, Int_t cruID){
    for(std::vector<DigitRow*>::iterator iterRow = mRows.begin(); iterRow != mRows.end(); iterRow++) {
      if((*iterRow) == nullptr) continue;
      (*iterRow)->FillOutputContainer(output, cruID, (*iterRow)->GetRow());
    }
}
