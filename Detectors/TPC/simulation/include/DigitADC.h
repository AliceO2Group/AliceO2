//
//  DigitADC.h
//  ALICEO2
//
//  Created by Markus Fasel on 26.03.15.
//
//

#ifndef _ALICEO2_DigitADC_
#define _ALICEO2_DigitADC_

#include "Rtypes.h"
#include "Digit.h"
#include <map>

class TClonesArray;

namespace AliceO2 {
    namespace TPC{
        
        class Digit;
        
        class DigitADC{
        public:
            DigitADC(Float_t charge);
            ~DigitADC();
            
            Float_t GetADC() {return mADC;}
                        
        private:
            Float_t mADC;
        };
    }
}

#endif
