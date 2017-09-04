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
/// \brief Implementation of the ITSMFT DigitContainer class
//
#include "ITSMFTBase/Digit.h"
#include "ITSMFTSimulation/DigitContainer.h"
#include "TRandom.h"
#include "FairLogger.h" // for LOG

using namespace o2::ITSMFT;

void DigitContainer::reset()
{
  for (Int_t i = 0; i < mChips.size(); i++){
    mChips[i].reset();
  }
}

Digit* DigitContainer::getDigit(Int_t chipID, UShort_t row, UShort_t col) { return mChips[chipID].getDigit(row, col); }

Digit* DigitContainer::addDigit(UShort_t chipID, UShort_t row, UShort_t col, Double_t charge, Double_t timestamp)
{
  return mChips[chipID].addDigit(chipID, row, col, charge, timestamp);
}

void DigitContainer::fillOutputContainer(TClonesArray* output)
{
  for (Int_t i = 0; i < mChips.size(); i++) {
    mChips[i].fillOutputContainer(output);
  }
}
