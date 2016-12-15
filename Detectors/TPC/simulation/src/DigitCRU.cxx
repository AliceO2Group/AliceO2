#include "TPCSimulation/DigitCRU.h"
#include "TPCSimulation/DigitTime.h"
#include "TPCBase/Mapper.h"

#include "FairLogger.h"
using namespace AliceO2::TPC;

#include <iostream>

DigitCRU::DigitCRU(Int_t cru):
mCRU(cru),
mTimeBins(500)
{}

DigitCRU::~DigitCRU() {
  for(auto &aTime : mTimeBins) {
    if(aTime == nullptr) continue;
    delete aTime;
  }
}

void DigitCRU::setDigit(Int_t timeBin, Int_t row, Int_t pad, Float_t charge) {
  //if time bin outside specified range, the range of the vector is extended by one full drift time.
  while(getSize() <= timeBin) {
    mTimeBins.resize(getSize() + 500);
  }
  
  DigitTime *result = mTimeBins[timeBin];
  if(result != nullptr) {
    mTimeBins[timeBin]->setDigit(mCRU, row, pad, charge);
  }
  else {
    const Mapper& mapper = Mapper::instance();
    mTimeBins[timeBin] = new DigitTime(timeBin, mapper.getPadRegionInfo(CRU(mCRU).region()).getNumberOfPadRows());
    mTimeBins[timeBin]->setDigit(mCRU, row, pad, charge);
  }
}

void DigitCRU::fillOutputContainer(TClonesArray *output, Int_t cru) {
  for(auto &aTime : mTimeBins) {
    if(aTime == nullptr) continue;
    aTime->fillOutputContainer(output, cru, aTime->getTimeBin());
  }
}
