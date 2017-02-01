/// \file DigitCRU.cxx
/// \author Andi Mathis, andreas.mathis@ph.tum.de

#include "TPCSimulation/DigitCRU.h"
#include "TPCSimulation/DigitTime.h"
#include "TPCBase/Mapper.h"

using namespace AliceO2::TPC;

DigitCRU::DigitCRU(Int_t cru)
  : mNTimeBins(500)
  , mCRU(cru)
  , mTimeBins(mNTimeBins)
{}

DigitCRU::~DigitCRU()
{
  for(auto &aTime : mTimeBins) {
    if(aTime == nullptr) continue;
    delete aTime;
  }
}

void DigitCRU::setDigit(Int_t eventID, Int_t trackID, Int_t timeBin, Int_t row, Int_t pad, Float_t charge)
{
  //if time bin outside specified range, the range of the vector is extended by one full drift time.
  while(getSize() <= timeBin) {
    mTimeBins.resize(getSize() + mNTimeBins);
  }
  
  DigitTime *result = mTimeBins[timeBin];
  if(result != nullptr) {
    mTimeBins[timeBin]->setDigit(eventID, trackID, mCRU, row, pad, charge);
  }
  else {
    const Mapper& mapper = Mapper::instance();
    mTimeBins[timeBin] = new DigitTime(timeBin, mapper.getPadRegionInfo(CRU(mCRU).region()).getNumberOfPadRows());
    mTimeBins[timeBin]->setDigit(eventID, trackID, mCRU, row, pad, charge);
  }
}

void DigitCRU::fillOutputContainer(TClonesArray *output, Int_t cru)
{
  for(auto &aTime : mTimeBins) {
    if(aTime == nullptr) continue;
    aTime->fillOutputContainer(output, cru, aTime->getTimeBin());
  }
}

void DigitCRU::fillOutputContainer(TClonesArray *output, Int_t cru, std::vector<CommonMode> &commonModeContainer)
{
  for(auto &aTime : mTimeBins) {
    if(aTime == nullptr) continue;
    aTime->fillOutputContainer(output, cru, aTime->getTimeBin(), commonModeContainer);
  }
}

void DigitCRU::processCommonMode(std::vector<CommonMode> & commonModeCRU, Int_t cru)
{
  for(auto &aTime : mTimeBins) {
    if(aTime == nullptr) continue;
    aTime->processCommonMode(cru, aTime->getTimeBin());
    CommonMode commonMode(cru, aTime->getTimeBin(), aTime->getTotalChargeTimeBin());
    commonModeCRU.emplace_back(commonMode);
  }
}
