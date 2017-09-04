// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file DigitChip.cxx
/// \brief Implementation of the ITSMFT DigitChip class

#include "ITSMFTBase/Digit.h"
#include "ITSMFTSimulation/DigitChip.h"

#include "FairLogger.h" // for LOG

#include "TClonesArray.h"

using namespace o2::ITSMFT;

Int_t DigitChip::sNumOfRows = 650;

DigitChip::DigitChip() {}

DigitChip::~DigitChip() { reset(); }

void DigitChip::reset()
{
  for (auto pixel : mPixels) {
    delete pixel.second;
  }
  mPixels.clear();
}

Digit* DigitChip::findDigit(Int_t idx)
{
  Digit* result = nullptr;
  auto digitentry = mPixels.find(idx);
  if (digitentry != mPixels.end()) {
    result = digitentry->second;
  }
  return result;
}

Digit* DigitChip::getDigit(UShort_t row, UShort_t col)
{
  Int_t idx = col * sNumOfRows + row;
  return findDigit(idx);
}

Digit* DigitChip::addDigit(UShort_t chipid, UShort_t row, UShort_t col, Double_t charge, Double_t timestamp)
{
  Int_t idx = col * sNumOfRows + row;

  Digit* digit = findDigit(idx);
  if (digit) {
    LOG(DEBUG) << "Adding charge to pixel..." << FairLogger::endl;
    charge += digit->getCharge();
    digit->setCharge(charge);
    return digit;
  }

  digit = new Digit(chipid, row, col, charge, timestamp);
  mPixels.insert(std::pair<Int_t, Digit*>(idx, digit));

  return digit;
}

void DigitChip::fillOutputContainer(TClonesArray* outputcont)
{
  TClonesArray& clref = *outputcont;
  for (auto digit : mPixels) {
    new (clref[clref.GetEntriesFast()]) Digit(*(digit.second));
  }
}

