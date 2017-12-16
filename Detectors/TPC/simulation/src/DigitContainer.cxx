// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file DigitContainer.cxx
/// \brief Implementation of the Digit Container
/// \author Andi Mathis, TU München, andreas.mathis@ph.tum.de

#include "TPCSimulation/DigitContainer.h"
#include "TPCBase/Mapper.h"
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
    mCRU[cru] = std::make_unique<DigitCRU>(cru, mCommonModeContainer);
    mCRU[cru]->setDigit(hitID, timeBin, row, pad, charge);
  }
  /// Take care of the common mode
  mCommonModeContainer.addDigit(cru, timeBin, charge);
}


void DigitContainer::fillOutputContainer(std::vector<o2::TPC::Digit> *output, o2::dataformats::MCTruthContainer<o2::MCCompLabel> &mcTruth,
                                         std::vector<o2::TPC::DigitMCMetaData> *debug, int eventTime, bool isContinuous)
{
  for(auto &aCRU : mCRU) {
    if(aCRU == nullptr) continue;
    aCRU->fillOutputContainer(output, mcTruth, debug, aCRU->getCRUID(), eventTime, isContinuous);
    if(!isContinuous) {
      aCRU->reset();
    }
  }
  mCommonModeContainer.cleanUp(eventTime, isContinuous);
}
