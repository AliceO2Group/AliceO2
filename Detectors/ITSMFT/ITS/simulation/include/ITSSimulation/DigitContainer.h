//
//  DigitContainer.h
//  ALICEO2
//

#ifndef _ALICEO2_ITS_DigitContainer_
#define _ALICEO2_ITS_DigitContainer_

#include "Rtypes.h"

class TClonesArray;

namespace AliceO2 {
namespace ITS {

class Digit;
class DigitChip;

class DigitContainer
{
  public:
    DigitContainer(Int_t nChips);
   ~DigitContainer();

    void Reset();

    Digit *AddDigit(
       UShort_t chipid, UShort_t row, UShort_t col,
       Double_t charge, Double_t timestamp
    );
    Digit *GetDigit(Int_t chipID, Int_t idx);

    void FillOutputContainer(TClonesArray *output);

  private:
    Int_t fnChips;
    DigitChip *fChips;
};
}
}

#endif /* defined(_ALICEO2_ITS_DigitContainer_) */
