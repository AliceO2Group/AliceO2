/// \file DigitContainer.h
/// \brief Definition of the ITS DigitContainer class

#ifndef ALICEO2_ITS_DIGITCONTAINER
#define ALICEO2_ITS_DIGITCONTAINER

#include <vector>

#include "ITSSimulation/DigitChip.h"
#include "Rtypes.h"

class TClonesArray;

namespace AliceO2
{
  namespace ITS
  {
    class Digit;

    class DigitContainer
    {
    public:
      DigitContainer() {}
      ~DigitContainer() {}
      void reset();
      void resize(Int_t n) { mChips.resize(n); }
      Digit* addDigit(UShort_t chipid, UShort_t row, UShort_t col, Double_t charge, Double_t timestamp);
      Digit* getDigit(Int_t chipID, UShort_t row, UShort_t col);

      void fillOutputContainer(TClonesArray* output);

    private:
      std::vector<DigitChip> mChips; ///< Vector of DigitChips
    };
  }
}

#endif /* ALICEO2_ITS_DIGITCONTAINER */
