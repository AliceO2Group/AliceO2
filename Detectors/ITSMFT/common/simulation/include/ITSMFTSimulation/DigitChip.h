// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See https://alice-o2.web.cern.ch/ for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file DigitChip.h
/// \brief Definition of the ITSMFT DigitChip class

#ifndef ALICEO2_ITSMFT_DIGITCHIP
#define ALICEO2_ITSMFT_DIGITCHIP

#include <map>
#include "Rtypes.h"

class TClonesArray;

namespace o2
{
  namespace ITSMFT
  {
    class Digit;
    class DigitChip
    {
    public:
      DigitChip();
      ~DigitChip();

      static void setNumberOfRows(Int_t nr) { sNumOfRows = nr; }
      void reset();

      o2::ITSMFT::Digit* getDigit(UShort_t row, UShort_t col);

      o2::ITSMFT::Digit* addDigit(UShort_t chipid, UShort_t row, UShort_t col, Double_t charge, Double_t timestamp);

      void fillOutputContainer(TClonesArray* outputcont);

    private:
      o2::ITSMFT::Digit* findDigit(Int_t idx);
      static Int_t sNumOfRows;         ///< Number of rows in the pixel matrix
      std::map<Int_t, o2::ITSMFT::Digit*> mPixels; ///< Map of fired pixels
    };
  }
}

#endif /* ALICEO2_ITSMFT_DIGITCHIP */
