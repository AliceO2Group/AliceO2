/// \file CommonMode.h
/// \brief Container class for the ADC values
/// \author Andi Mathis, andreas.mathis@ph.tum.de
#ifndef ALICEO2_TPC_CommonMode_H_
#define ALICEO2_TPC_CommonMode_H_

#include "TPCSimulation/Constants.h"

#include <vector>

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
    CommonMode(int cru, int timeBin, float charge);
      
    /// Destructor
    ~CommonMode();
      
    /// Get the ADC value
    /// @return ADC value
    float getCommonMode() {return mCommonMode;}
            
    /// Get the ADC value for a given CRU and Time bin from the Digits array
    /// @param summedChargesContainer Container containing the summed charges per pad and time bin
    /// @return container containing Common Mode objects with the proper amplitude
    float computeCommonMode(std::vector<CommonMode> & summedChargesContainer, std::vector<CommonMode> & commonModeContainer);
      
    /// Get the CRU ID
    /// @return CRU ID
    int getCRU() {return mCRU;}
      
    /// Get the Time bin
    /// @return Time Bin
    int getTimeBin() {return mTimeBin;}
      
  private:
    unsigned short      mCRU;
    unsigned short      mTimeBin;
    float               mCommonMode;
};
  
}
}

#endif // ALICEO2_TPC_CommonMode_H_
