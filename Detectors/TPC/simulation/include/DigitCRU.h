//
//  DigitCRU.h
//  ALICEO2
//
//  Created by Markus Fasel on 25.03.15.
//
//

#ifndef _ALICEO2_ITS_DigitCRU_
#define _ALICEO2_ITS_DigitCRU_

#include "Rtypes.h"

class TClonesArray;

namespace AliceO2 {
    namespace TPC {
        
        class Digit;
        class DigitADC;
        class DigitTime;
        class DigitPad;
        class DigitRow;
        
        class DigitCRU{
        public:
            DigitCRU(Int_t mCRUID, Int_t nrows);
            ~DigitCRU();
            
            void Reset();
            Int_t GetCRUID() {return mCRUID;}
            
            void SetDigit(Int_t row, Int_t pad, Int_t time, Float_t charge);
            
            void FillOutputContainer(TClonesArray *output, Int_t cruID);
            
        private:
            Int_t               mCRUID;           ///< CRU ID
            Int_t               mNRows;           ///< Number of rows in CRU
            std::vector <DigitRow*>  mRows;    // check whether there is something in the container before writing into it
        };
    }
}

#endif
