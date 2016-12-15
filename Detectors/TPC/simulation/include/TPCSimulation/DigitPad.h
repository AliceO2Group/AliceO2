/// \file DigitPad.h
/// \brief Digit container for the Digits
#ifndef _ALICEO2_TPC_DigitPad_
#define _ALICEO2_TPC_DigitPad_

#include "Rtypes.h"
#include "TPCSimulation/DigitADC.h"

class TClonesArray;

namespace AliceO2 {
  namespace TPC {
    
    /// \class DigitPad
    /// \brief Digit container class for the digits    
    
    class DigitPad{
    public:
      
      /// Constructor
      /// @param mPad Pad ID
      DigitPad(Int_t mPad);
      
      /// Destructor
      ~DigitPad();
      
      /// Resets the container
      void reset();
      
      /// Get the size of the container
      /// @return Size of the Time Bin container
      Int_t getSize() {return mADCCounts.size();}
      
      /// Get the Pad ID
      /// @return Pad ID
      Int_t getPad() {return mPad;}
      
      /// Add digit to the time bin container
      /// @param charge Charge of the digit
      void setDigit(Float_t charge);
      
      /// Fill output TClonesArray
      /// @param output Output container
      /// @param cru CRU ID
      /// @param timeBin Time bin
      /// @param row Row ID
      /// @param pad pad ID
      void fillOutputContainer(TClonesArray *output, Int_t cru, Int_t timeBin, Int_t row, Int_t pad);
      
    private:
      UChar_t                  mPad;
      std::vector <DigitADC>   mADCCounts;
    };
    
    inline 
    void DigitPad::setDigit(Float_t charge) {
      DigitADC digitAdc(charge);
      mADCCounts.emplace_back(digitAdc);
    }
    
    inline
    void DigitPad::reset() {
      mADCCounts.clear();
    }
    
    
  }
}

#endif
