/// \file DigitPad.h
/// \brief Digit container for the time bin Digits
#ifndef _ALICEO2_TPC_DigitPad_
#define _ALICEO2_TPC_DigitPad_

#include "Rtypes.h"

class TClonesArray;

namespace AliceO2 {
  namespace TPC {
    
    class Digit;
    class DigitADC;
    class DigitTime;
    class DigitPad;
    
    /// \class DigitPad
    /// \brief Digit container class for the time bin digits    
    
    class DigitPad{
    public:
      
      /// Constructor
      /// @param mPadID Pad ID
      DigitPad(Int_t mPadID);
      
      /// Destructor
      ~DigitPad();
      
      /// Resets the container
      void reset();
      
      /// Get the Pad ID
      /// @return Pad ID
      Int_t getPad() {return mPadID;}
      
      /// Add digit to the time bin container
      /// @param time Time bin of the digit
      /// @param charge Charge of the digit
      void setDigit(Int_t time, Float_t charge);
      
      /// Fill output TClonesArray
      /// @param output Output container
      /// @param cruID CRU ID
      /// @param rowID Row ID
      /// @param padID pad ID
      void fillOutputContainer(TClonesArray *output, Int_t cruID, Int_t rowID, Int_t padID);
      
    private:
      Int_t               mPadID;
      Int_t               mNTimeBins;
      std::vector <DigitTime*>  mTimeBins;
    };
  }
}

#endif
