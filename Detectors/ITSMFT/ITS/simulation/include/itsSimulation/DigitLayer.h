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

namespace AliceO2 {
namespace ITS {

class Digit;

class DigitStave;

class DigitLayer
{
  public:
    DigitLayer(Int_t fLayerID, Int_t nstaves);

    ~DigitLayer();

    void Reset();

    void SetDigit(Digit *digi, Int_t stave, Int_t pixel);

    Digit *FindDigit(Int_t stave, Int_t pixel);

    void FillOutputContainer(TClonesArray *output);

  private:
    Int_t fLayerID;           ///< Layer ID
    Int_t fNStaves;           ///< Number of staves in Layer
    DigitStave **fStaves;          ///< Container of staves
};
}
}

#endif
