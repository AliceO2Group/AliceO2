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
   namespace ITSMFT {
      class Digit;
   }
}

namespace AliceO2 {
namespace ITS {

class DigitStave
{
  public:
    DigitStave();

    ~DigitStave();

    void Reset();

    AliceO2::ITSMFT::Digit *FindDigit(Int_t pixel);

    void SetDigit(int pixel, AliceO2::ITSMFT::Digit *digi);

    void FillOutputContainer(TClonesArray *outputcont);

  private:
    std::map<int, AliceO2::ITSMFT::Digit *> mPixels;
};
}
}

#endif
