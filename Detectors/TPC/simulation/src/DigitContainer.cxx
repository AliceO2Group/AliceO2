/// \file DigitContainer.cxx
/// \author Andi Mathis, andreas.mathis@ph.tum.de

#include "TPCSimulation/DigitContainer.h"
#include "TPCSimulation/DigitCRU.h"
#include "TPCSimulation/CommonMode.h"
#include "TPCBase/Mapper.h"
#include "TPCBase/CRU.h"
#include <iostream>

using namespace AliceO2::TPC;

DigitContainer::DigitContainer()
  : mCRU()
{}

DigitContainer::~DigitContainer()
{
  std::fill(mCRU.begin(),mCRU.end(), nullptr);
}

void DigitContainer::addDigit(int eventID, int trackID, int cru, int timeBin, int row, int pad, float charge)
{
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


void DigitContainer::fillOutputContainer(TClonesArray *output)
{
  for(auto &aCRU : mCRU) {
    if(aCRU == nullptr) continue;
    aCRU->fillOutputContainer(output, aCRU->getCRUID());
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
