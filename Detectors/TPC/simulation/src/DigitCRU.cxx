#include "TPCSimulation/DigitCRU.h"
#include "TPCSimulation/DigitRow.h"
#include "TPCBase/Mapper.h"

#include "FairLogger.h"
using namespace AliceO2::TPC;

#include <iostream>

DigitCRU::DigitCRU(Int_t cruID, Int_t nrows):
mCRUID(cruID),
mNRows(nrows),
mRows(nrows)
{}

DigitCRU::~DigitCRU()
{
  for(auto iterRow = mRows.begin(); iterRow != mRows.end(); ++iterRow) {
    delete (*iterRow);
  }
}

void DigitCRU::setDigit(Int_t row, Int_t pad, Int_t time, Float_t charge)
{
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

void DigitCRU::reset()
{
  for(auto iterRow = mRows.begin(); iterRow != mRows.end(); ++iterRow) {
    if((*iterRow) == nullptr) continue;
    (*iterRow)->reset();
  }
}

void DigitCRU::fillOutputContainer(TClonesArray *output, Int_t cruID)
{
  for(auto iterRow = mRows.begin(); iterRow != mRows.end(); ++iterRow) {
    if((*iterRow) == nullptr) continue;
    (*iterRow)->fillOutputContainer(output, cruID, (*iterRow)->getRow());
  }
}
