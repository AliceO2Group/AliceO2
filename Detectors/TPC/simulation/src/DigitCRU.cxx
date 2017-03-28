/// \file DigitCRU.cxx
/// \author Andi Mathis, andreas.mathis@ph.tum.de

#include "TPCSimulation/DigitCRU.h"
#include "TPCSimulation/DigitTime.h"
#include "TPCBase/Mapper.h"

using namespace AliceO2::TPC;

DigitCRU::DigitCRU(int cru)
  : mNTimeBins(500)
  , mCRU(cru)
  , mTimeBins(mNTimeBins)
{}

DigitCRU::~DigitCRU()
{
  mNTimeBins = 0;
  mCRU = 0;
  mTimeBins.resize(0);
}

void DigitCRU::setDigit(int eventID, int trackID, int timeBin, int row, int pad, float charge)
{
  //if time bin outside specified range, the range of the vector is extended by one full drift time.
  while(getSize() <= timeBin) {
    mTimeBins.resize(getSize() + mNTimeBins);
  }
  
  DigitTime *result = mTimeBins[timeBin].get();
  if(result != nullptr) {
    mTimeBins[timeBin]->setDigit(eventID, trackID, mCRU, row, pad, charge);
  }
  else {
    const Mapper& mapper = Mapper::instance();
    mTimeBins[timeBin] = std::unique_ptr<DigitTime> (new DigitTime(timeBin, mapper.getPadRegionInfo(CRU(mCRU).region()).getNumberOfPadRows()));
    mTimeBins[timeBin]->setDigit(eventID, trackID, mCRU, row, pad, charge);
  }
}

void DigitCRU::fillOutputContainer(TClonesArray *output, int cru)
{
  for(auto &aTime : mTimeBins) {
    if(aTime == nullptr) continue;
    aTime->fillOutputContainer(output, cru, aTime->getTimeBin());
  }
}

void DigitCRU::fillOutputContainer(TClonesArray *output, int cru, std::vector<CommonMode> &commonModeContainer)
{
  for(auto &aTime : mTimeBins) {
    if(aTime == nullptr) continue;
    aTime->fillOutputContainer(output, cru, aTime->getTimeBin(), commonModeContainer);
  }
}

void DigitCRU::processCommonMode(std::vector<CommonMode> & commonModeCRU, int cru)
{
  for(auto &aTime : mTimeBins) {
    if(aTime == nullptr) continue;
    CommonMode commonMode(cru, aTime->getTimeBin(), aTime->getTotalChargeTimeBin());
    commonModeCRU.emplace_back(commonMode);
  }
}
