//
//  DigitLayer.h
//  ALICEO2
//
//  Created by Markus Fasel on 25.03.15.
//
//

#ifndef _ALICEO2_ITS_DigitLayer_
#define _ALICEO2_ITS_DigitLayer_

#include "Rtypes.h"

class TClonesArray;

namespace o2 {
  namespace ITSMFT {
    class Digit;
  }
}

namespace o2 {
namespace ITS {

class DigitStave;

class DigitLayer
{
  public:
    DigitLayer(Int_t fLayerID, Int_t nstaves);

    ~DigitLayer();

    void Reset();

    void SetDigit(o2::ITSMFT::Digit *digi, Int_t stave, Int_t pixel);

    o2::ITSMFT::Digit *FindDigit(Int_t stave, Int_t pixel);

    void FillOutputContainer(TClonesArray *output);

  private:
    Int_t mLayerID;           ///< Layer ID
    Int_t mNStaves;           ///< Number of staves in Layer
    DigitStave **mStaves;          ///< Container of staves
};
}
}

#endif
