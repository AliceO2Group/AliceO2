/// \file CommonMode.h
/// \brief Container class for the ADC values
#ifndef _ALICEO2_TPC_CommonModeCRU_
#define _ALICEO2_TPC_CommonModeCRU_

#include "Rtypes.h"
#include "TClonesArray.h"
#include "TPCSimulation/Constants.h"

namespace AliceO2 {
  namespace TPC {
    
    /// \class CommonMode
    /// \brief Container class for the Common Mode ADC values
    
    class CommonMode{
    public:
      
      /// Default constructor
      CommonMode();
      
      /// Constructor 
      /// @param cru CRU
      /// @param timeBin time bin
      /// @param charge Charge
      CommonMode(Int_t cru, Int_t timeBin, Float_t charge);
      
      /// Destructor
      ~CommonMode();
      
      /// Get the ADC value
      /// @return ADC value
      Float_t getCommonMode() {return mCommonMode;}
            
      /// Get the ADC value for a given CRU and Time bin from the Digits array
      /// @return ADC value
      Float_t computeCommonMode(std::vector<CommonMode> & summedChargesContainer, std::vector<CommonMode> & commonModeContainer);
      
      /// Get the CRU ID
      /// @return CRU ID
      Int_t getCRU() {return mCRU;}
      
      /// Get the Time bin
      /// @return Time Bin
      Int_t getTimeBin() {return mTimeBin;}
      
    private:
      UShort_t       mCRU;
      UShort_t       mTimeBin;
      Float_t        mCommonMode;
    };
  }
}

#endif
