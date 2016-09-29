/// \file DigitCRU.h
/// \brief Digit container for the Row Digits
#ifndef _ALICEO2_TPC_DigitCRU_
#define _ALICEO2_TPC_DigitCRU_

#include "Rtypes.h"

class TClonesArray;

namespace AliceO2 {
  namespace TPC {
    
    class Digit;
    class DigitADC;
    class DigitTime;
    class DigitPad;
    class DigitRow;
    
    /// \class DigitCRU
    /// \brief Digit container class for the row digits        
    
    class DigitCRU{
    public:
      
      /// Constructor
      /// @param mCRUID CRU ID
      /// @param nrows Number of rows in the CRU
      DigitCRU(Int_t mCRUID, Int_t nrows);
      
      /// Destructor
      ~DigitCRU();
      
      /// Resets the container
      void reset();
      
      /// Get the CRU ID
      /// @return CRU ID
      Int_t getCRUID() {return mCRUID;}
      
      /// Add digit to the row container
      /// @param row Pad row of digit
      /// @param pad Pad of digit
      /// @param time Time bin of the digit
      /// @param charge Charge of the digit
      void setDigit(Int_t row, Int_t pad, Int_t time, Float_t charge);
      
      /// Fill output TClonesArray
      /// @param output Output container
      /// @param cruID CRU ID
      void fillOutputContainer(TClonesArray *output, Int_t cruID);
      
    private:
      UShort_t                 mCRUID;           ///< CRU ID
      UChar_t                  mNRows;           ///< Number of rows in CRU
      std::vector <DigitRow*>  mRows;
    };
  }
}

#endif
