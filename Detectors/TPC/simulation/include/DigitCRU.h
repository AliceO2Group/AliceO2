/// \file DigitCRU.h
/// \brief Digit container for the Row Digits
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
            
            void reset();
            Int_t getCRUID() {return mCRUID;}
            
            void setDigit(Int_t row, Int_t pad, Int_t time, Float_t charge);
            
            void fillOutputContainer(TClonesArray *output, Int_t cruID);
            
        private:
            Int_t               mCRUID;           ///< CRU ID
            Int_t               mNRows;           ///< Number of rows in CRU
            std::vector <DigitRow*>  mRows;    // check whether there is something in the container before writing into it
        };
    }
}

#endif
