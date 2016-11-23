//
//  DigitChip.h
//  ALICEO2
//

#ifndef _ALICEO2_DigitChip_
#define _ALICEO2_DigitChip_

#include "Rtypes.h"
#include <map>

class TClonesArray;

namespace AliceO2 {
namespace ITS {

class Digit;

class DigitChip
{
  public:
    DigitChip();
   ~DigitChip();

    static void SetNRows(Int_t nr) { fnRows=nr; }
   
    void Reset();

    Digit *GetDigit(UShort_t row, UShort_t col);

    Digit *AddDigit(
       UShort_t chipid, UShort_t row, UShort_t col,
       Double_t charge, Double_t timestamp
    );
    
    void FillOutputContainer(TClonesArray *outputcont);

  private:
    Digit *FindDigit(Int_t idx);
    static Int_t fnRows;
    std::map<Int_t, Digit *> fPixels;
};
}
}

#endif
