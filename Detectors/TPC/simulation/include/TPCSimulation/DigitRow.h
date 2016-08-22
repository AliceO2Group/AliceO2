/// \file DigitRow.h
/// \brief Container class for the Row Digits
#ifndef _ALICEO2_TPC_DigitRow_
#define _ALICEO2_TPC_DigitRow_

#include "Rtypes.h"

class TClonesArray;

namespace AliceO2 {
    namespace TPC {
        
        class Digit;
        class DigitADC;
        class DigitTime;
        class DigitPad;
                
        /// \class DigitRow
        /// \brief Digit container class for the pad digits    
        
        class DigitRow{
        public:
          
            /// Constructor
            /// @param mRowID Row ID
            /// @param npads Number of pads in the row
            DigitRow(Int_t mRowID, Int_t npads);
            
            ///Destructor
            ~DigitRow();
            
            /// Resets the container
            void reset();
            
            /// Get the Row ID
            /// @return Row ID
            Int_t getRow() {return mRowID;}
            
            /// Add digit to the time bin container
            /// @param pad Pad of the digit
            /// @param time Time bin of the digit
            /// @param charge Charge of the digit
            void setDigit(Int_t pad, Int_t time, Float_t charge);
            
            /// Fill output TClonesArray
            /// @param output Output container
            /// @param cruID CRU ID
            /// @param rowID Row ID
            void fillOutputContainer(TClonesArray *output, Int_t cruID, Int_t rowID);
            
        private:
            Int_t               mRowID;
            Int_t               mNPads;
            std::vector<DigitPad*> mPads;
        };
    }
}

#endif
