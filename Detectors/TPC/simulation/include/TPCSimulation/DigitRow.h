/// \file DigitRow.h
/// \brief Container class for the Row Digits
#ifndef _ALICEO2_TPC_DigitRow_
#define _ALICEO2_TPC_DigitRow_

#include "Rtypes.h"
#include "TPCSimulation/DigitPad.h"

class TClonesArray;

namespace AliceO2 {
  namespace TPC {
    
    /// \class DigitRow
    /// \brief Digit container class for the pad digits    
    
    class DigitRow{
    public:
      
      /// Constructor
      /// @param mRow Row ID
      /// @param npads Number of pads in the row
      DigitRow(Int_t mRow, Int_t npads);
      
      ///Destructor
      ~DigitRow();
      
      /// Resets the container
      void reset();
      
      /// Get the size of the container
      /// @return Size of the pad container
      Int_t getSize() {return mPads.size();}
      
      /// Get the Row ID
      /// @return Row ID
      Int_t getRow() {return mRow;}
      
      /// Add digit to the pad container
      /// @param pad Pad of the digit
      /// @param charge Charge of the digit
      void setDigit(Int_t pad, Float_t charge);
      
      /// Fill output TClonesArray
      /// @param output Output container
      /// @param cruID CRU ID
      /// @param rowID Row ID
      void fillOutputContainer(TClonesArray *output, Int_t cru, Int_t timeBin, Int_t row);
      
    private:
      UChar_t                mRow;
      UChar_t                mNPads;
      std::vector<DigitPad*> mPads;
    };
    
    inline
    void DigitRow::reset() {
      for(auto &aPad : mPads) {
        if(aPad == nullptr) continue;
        aPad->reset();
      }
    }
    
    
  }
}

#endif
