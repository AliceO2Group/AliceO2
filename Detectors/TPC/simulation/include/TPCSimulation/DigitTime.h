/// \file DigitTime.h
/// \brief Container class for the ADC Digits
#ifndef _ALICEO2_TPC_DigitTime_
#define _ALICEO2_TPC_DigitTime_

#include "Rtypes.h"

class TClonesArray;

namespace AliceO2 {
    namespace TPC {
        
        class Digit;
        class DigitADC;
        class DigitTime;
        
        /// \class DigitTime
        /// \brief Digit container class for the ADC digits    
        
        class DigitTime{
        public:
            
            /// Constructor
            /// @param mTimeBin time bin
            DigitTime(Int_t mTimeBin);
            
            /// Destructor
            ~DigitTime();
            
            /// Resets the container            
            void reset();
            
            /// Get the time bin
            /// @return time bin          
            Int_t getTimeBin() {return mTimeBin;}
            
            /// Add digit to the ADC container
            /// @param charge Charge of the digit            
            void setDigit(Float_t charge);
            
            /// Fill output TClonesArray
            /// @param output Output container
            /// @param cruID CRU ID
            /// @param rowID Row ID
            /// @param padID Pad ID
            /// @param timeBin Time bin
            void fillOutputContainer(TClonesArray *output, Int_t cruID, Int_t rowID, Int_t padID, Int_t timeBin);
            
        private:
            Int_t               mTimeBin;
            Float_t             mADC;
            DigitADC            *digitAdc;
            std::vector <DigitADC*>  mADCCounts;
        };
    }
}

#endif
