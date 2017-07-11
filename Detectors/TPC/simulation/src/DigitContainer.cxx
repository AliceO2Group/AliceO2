// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See https://alice-o2.web.cern.ch/ for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

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

void DigitContainer::addDigit(size_t hitID, int cru, int timeBin, int row, int pad, float charge)
{
  /// Check whether the container at this spot already contains an entry
  DigitCRU *result = mCRU[cru].get();
  if(result != nullptr){
    mCRU[cru]->setDigit(hitID, timeBin, row, pad, charge);
  }
  else{
    const Mapper& mapper = Mapper::instance();
    mCRU[cru] = std::unique_ptr<DigitCRU> (new DigitCRU(cru));
    mCRU[cru]->setDigit(hitID, timeBin, row, pad, charge);
  }
}


void DigitContainer::fillOutputContainer(TClonesArray *output, int eventTime, bool isContinuous)
{
  for(auto &aCRU : mCRU) {
    if(aCRU == nullptr) continue;
    aCRU->fillOutputContainer(output, aCRU->getCRUID(), eventTime, isContinuous);
    if(!isContinuous) {
      aCRU->reset();
    }
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
