//
//  DigitChip.cxx
//  ALICEO2
//
#include "ITSSimulation/DigitChip.h"
#include "ITSSimulation/Digit.h"

#include "FairLogger.h"  // for LOG

#include "TClonesArray.h"

using namespace AliceO2::ITS;

DigitChip::DigitChip()
{
}

DigitChip::~DigitChip()
{
  Reset();
}

void DigitChip::Reset()
{
  for (auto pixel: fPixels) {
    delete pixel;
  }
  fPixels.clear();
}

Digit *DigitChip::GetDigit(Int_t idx)
{
  return fPixels[idx];
}

Digit *DigitChip::AddDigit(
       UShort_t chipid, UShort_t row, UShort_t col,
       Double_t charge, Double_t timestamp
)
{
  Digit *digit = new Digit(chipid,row,col,charge,timestamp);
  fPixels.push_back(digit);
  return digit;
}

void DigitChip::FillOutputContainer(TClonesArray *outputcont)
{
  TClonesArray &clref = *outputcont;
  for (auto digit: fPixels) {
    new(clref[clref.GetEntriesFast()]) Digit(*digit);
  }
}
