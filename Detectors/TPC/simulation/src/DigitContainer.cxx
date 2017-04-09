/// \file DigitContainer.cxx
/// \brief Implementation of the Digit Container
/// \author Andi Mathis, TU München, andreas.mathis@ph.tum.de

#include "TPCSimulation/DigitContainer.h"
#include "TPCSimulation/DigitCRU.h"
#include "TPCSimulation/CommonMode.h"
#include "TPCBase/Mapper.h"
#include "TPCBase/CRU.h"
#include <iostream>

using namespace o2::TPC;

void DigitContainer::addDigit(int eventID, int trackID, int cru, int timeBin, int row, int pad, float charge)
{
  /// Check whether the container at this spot already contains an entry
  DigitCRU *result = mCRU[cru].get();
  if(result != nullptr){
    mCRU[cru]->setDigit(eventID, trackID, timeBin, row, pad, charge);
  }
  else{
    const Mapper& mapper = Mapper::instance();
    mCRU[cru] = std::unique_ptr<DigitCRU> (new DigitCRU(cru));
    mCRU[cru]->setDigit(eventID, trackID, timeBin, row, pad, charge);
  }
}


void DigitContainer::fillOutputContainer(TClonesArray *output, int eventTime)
{
  for(auto &aCRU : mCRU) {
    if(aCRU == nullptr) continue;
    aCRU->fillOutputContainer(output, aCRU->getCRUID(), eventTime);
  }
}

void DigitContainer::fillOutputContainer(TClonesArray *output, std::vector<CommonMode> &commonModeContainer)
{
  for(auto &aCRU : mCRU) {
    if(aCRU == nullptr) continue;
    aCRU->fillOutputContainer(output, aCRU->getCRUID(), commonModeContainer);
  }
}


void DigitContainer::processCommonMode(std::vector<CommonMode> & commonModeContainer)
{
  std::vector<CommonMode> summedCharges(0);
  for(auto &aCRU : mCRU) {
    if(aCRU == nullptr) continue;
    aCRU->processCommonMode(summedCharges, aCRU->getCRUID());
  }
  
  CommonMode c;
  c.computeCommonMode(summedCharges, commonModeContainer);
}
