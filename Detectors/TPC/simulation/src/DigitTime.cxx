#include "TPCSimulation/DigitTime.h"
#include "TPCSimulation/DigitRow.h"
#include "TPCBase/Mapper.h"
#include "TClonesArray.h"
#include "FairLogger.h"
using namespace AliceO2::TPC;

DigitTime::DigitTime(Int_t timeBin, Int_t nrows):
mTimeBin(timeBin),
mNRows(nrows),
mRows(nrows)
{}

DigitTime::~DigitTime() {
  for(auto &aRow : mRows) {
    if(aRow == nullptr) continue;
    delete aRow;
  }
}

void DigitTime::setDigit(Int_t cru, Int_t row, Int_t pad, Float_t charge) {
  DigitRow *result = mRows[row];
  if(result != nullptr) {
    mRows[row]->setDigit(pad, charge);
  }
  else{
    const Mapper& mapper = Mapper::instance();
    mRows[row] = new DigitRow(row, mapper.getPadRegionInfo(CRU(cru).region()).getPadsInRowRegion(row));
    mRows[row]->setDigit(pad, charge);
  }
}

void DigitTime::fillOutputContainer(TClonesArray *output, Int_t cru, Int_t timeBin) {
  for(auto &aRow : mRows) {
    if(aRow == nullptr) continue;
    aRow->fillOutputContainer(output, cru, timeBin, aRow->getRow());
  }
}