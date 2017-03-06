/// \file DigitTime.cxx
/// \author Andi Mathis, andreas.mathis@ph.tum.de

#include "TPCSimulation/DigitTime.h"
#include "TPCSimulation/DigitRow.h"
#include "TPCBase/Mapper.h"

using namespace AliceO2::TPC;

DigitTime::DigitTime(int timeBin, int nrows)
  : mTotalChargeTimeBin(0.)
  , mTimeBin(timeBin)
  , mRows(nrows)
{}

DigitTime::~DigitTime()
{
  mTotalChargeTimeBin = 0;
  mTimeBin = 0;
  mRows.resize(0);
}

void DigitTime::setDigit(int eventID, int trackID, int cru, int row, int pad, float charge)
{
  DigitRow *result = mRows[row].get();
  if(result != nullptr) {
    mRows[row]->setDigit(eventID, trackID, pad, charge);
  }
  else{
    const Mapper& mapper = Mapper::instance();
    mRows[row] = std::unique_ptr<DigitRow> (new DigitRow(row, mapper.getPadRegionInfo(CRU(cru).region()).getPadsInRowRegion(row)));
    mRows[row]->setDigit(eventID, trackID, pad, charge);
  }
  mTotalChargeTimeBin+=charge;
}

void DigitTime::fillOutputContainer(TClonesArray *output, int cru, int timeBin)
{
  for(auto &aRow : mRows) {
    if(aRow == nullptr) continue;
    aRow->fillOutputContainer(output, cru, timeBin, aRow->getRow());
  }
}

void DigitTime::fillOutputContainer(TClonesArray *output, int cru, int timeBin, std::vector<CommonMode> &commonModeContainer)
{
  float commonMode =0;
  for (auto &aCommonMode :commonModeContainer){
    if(aCommonMode.getCRU() == cru && aCommonMode.getTimeBin() == timeBin) {
      commonMode = aCommonMode.getCommonMode();
      break;
    }
  }

  for(auto &aRow : mRows) {
    if(aRow == nullptr) continue;
    aRow->fillOutputContainer(output, cru, timeBin, aRow->getRow(), commonMode);
  }
}