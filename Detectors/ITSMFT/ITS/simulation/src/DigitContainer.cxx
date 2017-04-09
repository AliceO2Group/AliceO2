/// \file DigitContainer.cxx
/// \brief Implementation of the ITS DigitContainer class
//
#include "ITSSimulation/DigitContainer.h"
#include "ITSMFTBase/Digit.h"

#include "FairLogger.h" // for LOG

using o2::ITSMFT::Digit;
using namespace o2::ITS;

void DigitContainer::reset()
{
  for (Int_t i = 0; i < mChips.size(); i++)
    mChips[i].reset();
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
