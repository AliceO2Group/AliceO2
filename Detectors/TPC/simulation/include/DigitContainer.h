#ifndef _ALICEO2_DigitContainer_
#define _ALICEO2_DigitContainer_

#include "Digit.h"
#include "DigitCRU.h"
#include "Rtypes.h"
#include <map>

class TClonesArray;

namespace AliceO2 {
    namespace TPC{
        class Digit;
        class DigitPad;
        class DigitRow;
        class DigitCRU;
        
        class DigitContainer{
        public:
            DigitContainer();
            ~DigitContainer();
            
            void Reset();
            
            void AddDigit(Int_t cru, Int_t row, Int_t pad, Int_t time, Float_t charge);
            void FillOutputContainer(TClonesArray *outputcont);
            
        private:
          Int_t mNCRU;
          std::vector<DigitCRU*> mCRU;
        };
    }
}

#endif
