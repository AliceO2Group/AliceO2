//
//  DigitContainer.h
//  ALICEO2
//

#ifndef _ALICEO2_ITS_DigitContainer_
#define _ALICEO2_ITS_DigitContainer_

#include <vector>

#include "Rtypes.h"
#include "ITSSimulation/DigitChip.h"

class TClonesArray;

namespace AliceO2 {
namespace ITS {

class Digit;

class DigitContainer
{
  public:
    DigitContainer() {}
   ~DigitContainer() {}

    void Reset();
    void Resize(Int_t n) {fChips.resize(n);}
    
    Digit *AddDigit(
       UShort_t chipid, UShort_t row, UShort_t col,
       Double_t charge, Double_t timestamp
    );
    Digit *GetDigit(Int_t chipID, UShort_t row, UShort_t col);

    void FillOutputContainer(TClonesArray *output);

  private:
    std::vector<DigitChip> fChips;
};
}
}

#endif /* defined(_ALICEO2_ITS_DigitContainer_) */
