/// \file DigitContainer.h
/// \brief Container class for the CRU Digits
#ifndef _ALICEO2_TPC_DigitContainer_
#define _ALICEO2_TPC_DigitContainer_

#include "TPCSimulation/Digit.h"
#include "TPCSimulation/DigitCRU.h"
#include "Rtypes.h"
#include <map>

class TClonesArray;

namespace AliceO2 {
  namespace TPC {
    class Digit;
    class DigitPad;
    class DigitRow;
    class DigitCRU;
    
    /// \class DigitContainer
    /// \brief Digit container class
    
    class DigitContainer{
    public:
      
      /// Default constructor
      DigitContainer();
      
      /// Destructor
      ~DigitContainer();
      
      void reset();
      
      /// Add digit to the container
      /// @param cru CRU of the digit
      /// @param row Pad row of digit
      /// @param pad Pad of digit
      /// @param time Time bin of the digit
      /// @param charge Charge of the digit
      void addDigit(Int_t cru, Int_t row, Int_t pad, Int_t time, Float_t charge);
      
      /// Fill output TClonesArray
      /// @param outputcont Output container
      void fillOutputContainer(TClonesArray *outputcont);
      
    private:
      Int_t mNCRU;
      std::vector<DigitCRU*> mCRU;
    };
  }
}

#endif
