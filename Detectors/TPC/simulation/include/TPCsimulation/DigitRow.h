/// \file DigitPad.h
/// \brief Container class for the Pad Digits
#ifndef _ALICEO2_ITS_DigitRow_
#define _ALICEO2_ITS_DigitRow_

#include "Rtypes.h"

class TClonesArray;

namespace AliceO2 {
    namespace TPC {
        
        class Digit;
        class DigitADC;
        class DigitTime;
        class DigitPad;
        
        class DigitRow{
        public:
            DigitRow(Int_t mRowID, Int_t npads);
            ~DigitRow();
            
            void reset();
            
            Int_t getRow() {return mRowID;}
            
            void setDigit(Int_t pad, Int_t time, Float_t charge);
            
            void fillOutputContainer(TClonesArray *output, Int_t cruID, Int_t rowID);
            
        private:
            Int_t               mRowID;           ///< Layer ID
            Int_t               mNPads;           ///< Number of staves in Layer
            std::vector<DigitPad*> mPads;
        };
    }
}

#endif
