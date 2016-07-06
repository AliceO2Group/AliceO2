#ifndef _ALICEO2_ITS_DigitPad_
#define _ALICEO2_ITS_DigitPad_

#include "Rtypes.h"

class TClonesArray;

namespace AliceO2 {
    namespace TPC {
        
        class Digit;
        class DigitADC;
        class DigitTime;
        class DigitPad;
        
        class DigitPad{
        public:
            DigitPad(Int_t mPadID, Int_t nTimeBins);
            ~DigitPad();
            
            void Reset();
            
            Int_t GetPad() {return mPadID;}
            
            void SetDigit(Int_t time, Float_t charge);
            
            void FillOutputContainer(TClonesArray *output, Int_t cruID, Int_t rowID, Int_t padID);
            
        private:
            Int_t               mPadID;           ///< Layer ID
            Int_t               mNTimeBins;           ///< Number of staves in Layer
//             DigitTime           **mTimeBins;          ///< Container of staves
            std::vector <DigitTime*>  mTimeBins;
        };
    }
}

#endif
