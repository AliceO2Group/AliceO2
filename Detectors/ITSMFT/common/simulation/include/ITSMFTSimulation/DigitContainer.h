/// \file DigitContainer.h
/// \brief Definition of the ITSMFT DigitContainer class

#ifndef ALICEO2_ITSMFT_DIGITCONTAINER
#define ALICEO2_ITSMFT_DIGITCONTAINER

#include <vector>

#include "Rtypes.h"
#include "ITSMFTSimulation/DigitChip.h"
#include "ITSMFTBase/SegmentationPixel.h"

class TClonesArray;

namespace o2
{
  namespace ITSMFT
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

#endif /* ALICEO2_ITSMFT_DIGITCONTAINER */
