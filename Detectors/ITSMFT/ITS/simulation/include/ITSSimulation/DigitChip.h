//
//  DigitChip.h
//  ALICEO2
//

#ifndef _ALICEO2_DigitChip_
#define _ALICEO2_DigitChip_

#include "Rtypes.h"
#include <vector>

class TClonesArray;

namespace AliceO2 {
namespace ITS {

class Digit;

class DigitChip
{
  public:
    DigitChip();
   ~DigitChip();

    void Reset();

    Digit *GetDigit(Int_t idx);

    Digit *AddDigit(
       UShort_t chipid, UShort_t row, UShort_t col,
       Double_t charge, Double_t timestamp
    );
    
    void FillOutputContainer(TClonesArray *outputcont);

  private:
    std::vector<Digit *> fPixels;
};
}
}

#endif
