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
  for (Int_t i = 0; i < mChips.size(); i++)
  mChips[i].reset();
}

void DigitContainer::addNoise(Double_t mean, const SegmentationPixel* seg) {
  UInt_t row = 0;
  UInt_t col = 0;
  Int_t nhits = 0;
  for (size_t chip = 0; chip < mChips.size(); ++chip) {
    nhits = gRandom->Poisson(mean);
    for (Int_t i = 0; i < nhits; ++i) {
      row = gRandom->Integer(seg->getNumberOfRows());
      col = gRandom->Integer(seg->getNumberOfColumns());
      Digit *noiseD = mChips[chip].addDigit(chip, row, col, 0., 0.);
      noiseD->setLabel(0, -1);
    }
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
