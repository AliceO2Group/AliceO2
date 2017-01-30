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
  : mCRU(CRU::MaxCRU)
{}

DigitContainer::~DigitContainer()
{
  for(auto &aCRU : mCRU) {
    if(aCRU == nullptr) continue;
    delete aCRU;
  }
}

void DigitContainer::addDigit(Int_t cru, Int_t timeBin, Int_t row, Int_t pad, Float_t charge)
{
  DigitCRU *result = mCRU[cru];
  if(result != nullptr){
    mCRU[cru]->setDigit(timeBin, row, pad, charge);
  }
  else{
    const Mapper& mapper = Mapper::instance();
    mCRU[cru] = new DigitCRU(cru);
    mCRU[cru]->setDigit(timeBin, row, pad, charge);
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
