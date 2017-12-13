// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file DigitRow.cxx
/// \brief Implementation of the Row container
/// \author Andi Mathis, TU München, andreas.mathis@ph.tum.de

#include "TPCSimulation/DigitRow.h"
#include "TPCSimulation/DigitPad.h"

using namespace o2::TPC;

void DigitRow::setDigit(size_t hitID, int pad, float charge)
{
  /// Check whether the container at this spot already contains an entry
  DigitPad *result =  mPads[pad].get();
  if(result != nullptr) {
    mPads[pad]->setDigit(hitID, charge);
  }
  else{
    mPads[pad] = std::make_unique<DigitPad> (pad);
    mPads[pad]->setDigit(hitID, charge);
  }
}

void DigitRow::fillOutputContainer(std::vector<o2::TPC::Digit> *output, o2::dataformats::MCTruthContainer<o2::MCCompLabel> &mcTruth,
                                   std::vector<o2::TPC::DigitMCMetaData> *debug, int cru, int timeBin, int row, float commonMode)
{
  for(auto &aPad : mPads) {
    if(aPad == nullptr) continue;
    aPad->fillOutputContainer(output, mcTruth, debug, cru, timeBin, row, aPad->getPad(), commonMode);
  }
}
