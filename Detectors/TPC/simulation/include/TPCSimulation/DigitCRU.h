/// \file DigitCRU.h
/// \brief Digit container for the Time bin Digits
#ifndef _ALICEO2_TPC_DigitCRU_
#define _ALICEO2_TPC_DigitCRU_

#include "Rtypes.h"
#include "DigitTime.h"

class TClonesArray;

namespace AliceO2 {
  namespace TPC {
        
    /// \class DigitCRU
    /// \brief Digit container class for the time bin digits        
    
    class DigitCRU{
    public:
      
      /// Constructor
      /// @param mCRU CRU ID
      DigitCRU(Int_t mCRU);
      
      /// Destructor
      ~DigitCRU();
      
      /// Resets the container
      void reset();
      
      /// Get the size of the container
      /// @return Size of the row container
      Int_t getSize() {return mTimeBins.size();}
      
      /// Get the CRU ID
      /// @return CRU ID
      Int_t getCRUID() {return mCRU;}
      
      /// Add digit to the row container
      /// @param row Pad row of digit
      /// @param pad Pad of digit
      /// @param timeBin Time bin of the digit
      /// @param charge Charge of the digit
      void setDigit(Int_t timeBin, Int_t row, Int_t pad, Float_t charge);
      
      /// Fill output TClonesArray
      /// @param output Output container
      /// @param cruID CRU ID
      void fillOutputContainer(TClonesArray *output, Int_t cru);
      
    private:
      UShort_t                 mCRU;
      Int_t                    mNTimeBins;
      std::vector <DigitTime*> mTimeBins;
    };
    
    
    inline 
    void DigitCRU::reset() {
      for(auto &aTime : mTimeBins) {
        if(aTime == nullptr) continue;
        aTime->reset();
      }
    }
    
    
  }
}

#endif
