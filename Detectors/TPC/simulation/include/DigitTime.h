#ifndef _ALICEO2_ITS_DigitTime_
#define _ALICEO2_ITS_DigitTime_

#include "Rtypes.h"

class TClonesArray;

namespace AliceO2 {
    namespace TPC {
        
        class Digit;
        class DigitADC;
        class DigitTime;
        
        class DigitTime{
        public:
            DigitTime(Int_t mTimeBin);
            ~DigitTime();
            
            void Reset();
            
            Int_t GetTimeBin() {return mTimeBin;}
            
            void SetDigit(Float_t charge);
            
            void FillOutputContainer(TClonesArray *output, Int_t cruID, Int_t rowID, Int_t padID, Int_t timeBin);
            
        private:
            Int_t               mTimeBin;
            Float_t             mCharge;
            DigitADC            *digitAdc;
            std::vector <DigitADC*>  mADCCounts;
        };
    }
}

#endif
