//
//  DigitStave.h
//  ALICEO2
//
//  Created by Markus Fasel on 26.03.15.
//
//

#ifndef _ALICEO2_DigitStave_
#define _ALICEO2_DigitStave_

#include "Rtypes.h"
#include <map>

class TClonesArray;

namespace AliceO2 {
namespace ITS {

class Digit;

class DigitStave
{
  public:
    DigitStave();

    ~DigitStave();

    void Reset();

    Digit *FindDigit(Int_t pixel);

    void SetDigit(int pixel, Digit *digi);

    void FillOutputContainer(TClonesArray *outputcont);

  private:
    std::map<int, Digit *> fPixels;
};
}
}

#endif
