// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See https://alice-o2.web.cern.ch/ for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

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
