/// \file DigitTime.h
/// \brief Container class for the Row Digits
#ifndef _ALICEO2_TPC_DigitTime_
#define _ALICEO2_TPC_DigitTime_

#include "Rtypes.h"
#include "TPCSimulation/DigitRow.h"
#include "TPCSimulation/CommonMode.h"

class TClonesArray;

namespace AliceO2 {
  namespace TPC {
    
    /// \class DigitTime
    /// \brief Digit container class for the Row digits    
    
    class DigitTime{
    public:

      /// Constructor
      /// @param mTimeBin time bin
      /// @param npads Number of pads in the row
      DigitTime(Int_t mTimeBin, Int_t nrows);

      /// Destructor
      ~DigitTime();

      /// Resets the container            
      void reset();

      /// Get the size of the container
      /// @return Size of the Row container
      Int_t getSize() {return mRows.size();}

      /// Get the number of entries in the container
      /// @return Number of entries in the Row container
      Int_t getNentries();

      /// Get the time bin
      /// @return time bin          
      Int_t getTimeBin() {return mTimeBin;}

      /// Get the accumulated charge in one time bin
      /// @return Accumulated charge in one time bin
      Float_t getTotalChargeTimeBin() {return mTotalChargeTimeBin;}

      /// Add digit to the row container
      /// @param row Pad row of digit
      /// @param pad Pad of digit
      /// @param charge Charge of the digit
      void setDigit(Int_t cru, Int_t row, Int_t pad, Float_t charge);

      /// Fill output TClonesArray
      /// @param output Output container
      /// @param cru CRU
      /// @param timeBin Time bin
      void fillOutputContainer(TClonesArray *output, Int_t cru, Int_t timeBin);

      /// Fill output TClonesArray
      /// @param output Output container
      /// @param cru CRU
      /// @param timeBin Time bin
      void fillOutputContainer(TClonesArray *output, Int_t cru, Int_t timeBin, std::vector<CommonMode> commonModeContainer);

      /// Process Common Mode Information
      /// @param output Output container
      /// @param cru CRU
      /// @param timeBin Time bin
      void processCommonMode(Int_t cru, Int_t timeBin);

    private:
      UShort_t                mTimeBin;
      UChar_t                 mNRows;
      Float_t                 mTotalChargeTimeBin;
      std::vector <DigitRow*> mRows;
    };

    inline    
    void DigitTime::reset() {  
      for(auto &aRow : mRows) {
        if(aRow == nullptr) continue;
        aRow->reset();
      }
      mRows.clear();
    }

    inline    
    Int_t DigitTime::getNentries() {
      Int_t counter = 0;
      for(auto &aRow : mRows) {
        if(aRow == nullptr) continue;
        ++ counter;
      }
      return counter;
    }
  }
}

#endif
