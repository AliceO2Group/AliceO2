/// \file DigitContainer.h
/// \brief Definition of the ITS DigitContainer class

#ifndef ALICEO2_ITS_DIGITCONTAINER
#define ALICEO2_ITS_DIGITCONTAINER

#include <vector>

#include "ITSSimulation/DigitChip.h"
#include "Rtypes.h"

class TClonesArray;

namespace o2
{
  namespace ITSMFT
  {
    class Digit;
  }
}

namespace o2
{
  namespace ITS
  {
    class DigitContainer
    {
    public:
      DigitContainer() {}
      ~DigitContainer() = default;
      void reset();
      void resize(Int_t n) { mChips.resize(n); }
      o2::ITSMFT::Digit* addDigit(UShort_t chipid, UShort_t row, UShort_t col, Double_t charge, Double_t timestamp);
      o2::ITSMFT::Digit* getDigit(Int_t chipID, UShort_t row, UShort_t col);

      void fillOutputContainer(TClonesArray* output);

    private:
      std::vector<DigitChip> mChips; ///< Vector of DigitChips
    };
  }
}

#endif /* ALICEO2_ITS_DIGITCONTAINER */
