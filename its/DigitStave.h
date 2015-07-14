//
//  DigitStave.h
//  ALICEO2
//
//  Created by Markus Fasel on 26.03.15.
//
//

#ifndef _ALICEO2_DigitStave_
#define _ALICEO2_DigitStave_

#include "RTypes.h"
#include <map>

namespace AliceO2 {
    namespace ITS{
        
        class Digit;
        
        class DigitStave{
        public:
            DigitStave(Int_t npixel);
            ~DigitStave();
            
            void Reset();
            
            Digit *FindDigit(Int_t pixel);
            void SetDigit(int pixel, Digit *digi);
            
        private:
            Int_t                       fNumberOfPixels;
            std::map<int, Digit*>       fPixels;
        };
    }
}

#endif
