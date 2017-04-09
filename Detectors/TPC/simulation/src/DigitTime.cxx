/// \file DigitTime.cxx
/// \brief Implementation of the Time Bin container
/// \author Andi Mathis, TU München, andreas.mathis@ph.tum.de

#include "TPCSimulation/DigitTime.h"
#include "TPCSimulation/DigitRow.h"
#include "TPCBase/Mapper.h"

using namespace o2::TPC;

void DigitTime::setDigit(int eventID, int trackID, int cru, int row, int pad, float charge)
{
  /// Check whether the container at this spot already contains an entry
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
