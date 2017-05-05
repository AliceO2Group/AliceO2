/// \file DigitContainer.h
/// \brief Container of digits
/// \author bogdan.vulpescu@cern.ch 
/// \date 03/05/2017

#ifndef ALICEO2_MFT_DIGITCONTAINER
#define ALICEO2_MFT_DIGITCONTAINER

namespace o2
{
  namespace ITSMFT
  {
    class Digit;
  }
}

namespace o2
{
  namespace MFT
  {
    class DigitContainer
    {
      
    public:

      DigitContainer() {}
      ~DigitContainer() = default;

      void reset();
      o2::ITSMFT::Digit* addDigit();
      o2::ITSMFT::Digit* getDigit();
      
      void fillOutputContainer(TClonesArray* output);

    private:

      ClassDef(DigitContainer,1);

    };
  }
}

#endif
      
