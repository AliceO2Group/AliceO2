/// \file DigitADC.h
/// \brief Container class for the ADC values
#ifndef _ALICEO2_TPC_DigitADC_
#define _ALICEO2_TPC_DigitADC_

#include "Rtypes.h"
#include "TPCSimulation/Digit.h"
#include <map>

class TClonesArray;

namespace AliceO2 {
    namespace TPC{

        class Digit;
        
        /// \class DigitADC
        /// \brief Digit container class for the ADC values

        class DigitADC{
        public:
          
            /// Constructor 
            /// @param charge Charge
            DigitADC(Float_t charge);
            
            /// Destructor
            ~DigitADC();

            /// Get the ADC value
            /// @return ADC value
            Float_t getADC() {return mADC;}

        private:
            Float_t mADC;
        };
    }
}

#endif
