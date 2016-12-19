//
//  DigitChip.cxx
//  ALICEO2
//
#include "ITSSimulation/DigitChip.h"
#include "ITSBase/Digit.h"

#include "FairLogger.h"  // for LOG

#include "TClonesArray.h"

using namespace AliceO2::ITS;

Int_t DigitChip::fnRows=650;

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
    delete pixel.second;
  }
  fPixels.clear();
}

Digit *DigitChip::FindDigit(Int_t idx)
{
  Digit *result = nullptr;
  auto digitentry = fPixels.find(idx);
  if (digitentry != fPixels.end()) {
    result = digitentry->second;
  }
  return result;
}

Digit *DigitChip::GetDigit(UShort_t row, UShort_t col) {
  Int_t idx = col*fnRows + row;
  return FindDigit(idx); 
}

Digit *DigitChip::AddDigit(
       UShort_t chipid, UShort_t row, UShort_t col,
       Double_t charge, Double_t timestamp
)
{
  Int_t idx = col*fnRows + row;

  Digit *digit=FindDigit(idx);
  if (digit) {
    LOG(DEBUG) << "Adding charge to pixel..." << FairLogger::endl;
    charge += digit->getCharge();
    delete digit;
  }
  
  digit = new Digit(chipid,row,col,charge,timestamp);
  fPixels.insert(std::pair<Int_t, Digit *>(idx,digit));

  return digit;
}

void DigitChip::FillOutputContainer(TClonesArray *outputcont)
{
  TClonesArray &clref = *outputcont;
  for (auto digit: fPixels) {
    new(clref[clref.GetEntriesFast()]) Digit(*(digit.second));
  }
}
