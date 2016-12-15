//
//  DigitContainer.cxx
//  ALICEO2
//
#include "ITSSimulation/DigitContainer.h"
#include "ITSBase/Digit.h"

#include "FairLogger.h"           // for LOG

using namespace AliceO2::ITS;

void DigitContainer::Reset()
{
  for (Int_t i = 0; i < fChips.size(); i++) fChips[i].Reset();
}

Digit *DigitContainer::GetDigit(Int_t chipID, UShort_t row, UShort_t col)
{
  return fChips[chipID].GetDigit(row,col);
}

Digit *DigitContainer::AddDigit(
       UShort_t chipID, UShort_t row, UShort_t col,
       Double_t charge, Double_t timestamp
)
{
  return fChips[chipID].AddDigit(chipID,row,col,charge,timestamp);
}

void DigitContainer::FillOutputContainer(TClonesArray *output)
{
  for (Int_t i = 0; i < fChips.size(); i++) {
    fChips[i].FillOutputContainer(output);
  }
}
