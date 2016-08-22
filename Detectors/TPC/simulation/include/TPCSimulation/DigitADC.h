/// \file DigitADC.h
/// \brief Container class for the ADC values
#ifndef _ALICEO2_DigitADC_
#define _ALICEO2_DigitADC_

#include "Rtypes.h"
#include "TPCSimulation/Digit.h"
#include <map>

class TClonesArray;

namespace AliceO2 {
    namespace TPC{

        class Digit;

        class DigitADC{
        public:
            DigitADC(Float_t charge);
            ~DigitADC();

            Float_t getADC() {return mADC;}

        private:
            Float_t mADC;
        };
    }
}

#endif
