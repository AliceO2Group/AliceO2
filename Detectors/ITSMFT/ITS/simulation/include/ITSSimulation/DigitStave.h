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

namespace o2 {
   namespace ITSMFT {
      class Digit;
   }
}

namespace o2 {
namespace ITS {

class DigitStave
{
  public:
    DigitStave();

    ~DigitStave();

    void Reset();

    o2::ITSMFT::Digit *FindDigit(Int_t pixel);

    void SetDigit(int pixel, o2::ITSMFT::Digit *digi);

    void FillOutputContainer(TClonesArray *outputcont);

  private:
    std::map<int, o2::ITSMFT::Digit *> mPixels;
};
}
}

#endif
