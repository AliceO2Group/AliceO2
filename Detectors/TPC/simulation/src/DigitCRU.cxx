/// \file DigitCRU.cxx
/// \brief Implementation of the CRU container
/// \author Andi Mathis, TU München, andreas.mathis@ph.tum.de

#include "TPCSimulation/DigitCRU.h"
#include "TPCSimulation/DigitTime.h"
#include "TPCBase/Mapper.h"

#include "FairLogger.h"

using namespace o2::TPC;

void DigitCRU::setDigit(int eventID, int trackID, int timeBin, int row, int pad, float charge)
{
  mEffectiveTimeBin = timeBin - mFirstTimeBin;
  if(mEffectiveTimeBin < 0.) {
    LOG(FATAL) << "TPC DigitCRU buffer misaligned ";
    LOG(DEBUG) << "for Event " << eventID << " CRU " <<mCRU << " TimeBin " << timeBin << " First TimeBin " << mFirstTimeBin << " Row " << row << " Pad " << pad;
    LOG(FATAL) << FairLogger::endl;
    return;
  }
  /// If time bin outside specified range, the range of the vector is extended by one full drift time.
  while(getSize() <= mEffectiveTimeBin) {
    mTimeBins.resize(getSize() + mNTimeBins);
  }
  /// Check whether the container at this spot already contains an entry
  DigitTime *result = mTimeBins[mEffectiveTimeBin].get();
  if(result != nullptr) {
    mTimeBins[mEffectiveTimeBin]->setDigit(eventID, trackID, mCRU, row, pad, charge);
  }
  else {
    const Mapper& mapper = Mapper::instance();
    mTimeBins[mEffectiveTimeBin] = std::unique_ptr<DigitTime> (new DigitTime(timeBin, mapper.getPadRegionInfo(CRU(mCRU).region()).getNumberOfPadRows()));
    mTimeBins[mEffectiveTimeBin]->setDigit(eventID, trackID, mCRU, row, pad, charge);
  }
}

void DigitCRU::fillOutputContainer(TClonesArray *output, int cru, int eventTime)
{
  int nProcessedTimeBins = 0;
  for(auto &aTime : mTimeBins) {
    if(nProcessedTimeBins + mFirstTimeBin < eventTime) {
      ++nProcessedTimeBins;
      if(aTime == nullptr) continue;
      aTime->fillOutputContainer(output, cru, aTime->getTimeBin());
    }
    else break;
  }
  if(nProcessedTimeBins > 0) {
    mFirstTimeBin += nProcessedTimeBins;
    while(nProcessedTimeBins--) {
      mTimeBins.pop_front();
    }
  }
  mTimeBinLastEvent = eventTime;
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
