/// \file CommonMode.h
/// \brief Definition of the Common Mode computation
/// \author Andi Mathis, TU München, andreas.mathis@ph.tum.de

#ifndef ALICEO2_TPC_CommonMode_H_
#define ALICEO2_TPC_CommonMode_H_

#include "TPCSimulation/Constants.h"

#include <vector>

namespace o2 {
namespace TPC {
    
/// \class CommonMode
/// This class computes the Common Mode value for a given ROC and makes it accessible for the Digits
    
class CommonMode{
  public:
      
    /// Default constructor
    CommonMode();
      
    /// Constructor 
    /// \param cru CRU
    /// \param timeBin time bin
    /// \param charge Charge
    CommonMode(int cru, int timeBin, float charge);
      
    /// Destructor
    ~CommonMode();
      
    /// Get the ADC value
    /// \return ADC value
    float getCommonMode() const {return mCommonMode;}
            
    /// Get the ADC value for a given CRU and Time bin from the Digits array
    /// \param summedChargesContainer Container containing the summed charges per pad and time bin
    /// \return container containing Common Mode objects with the proper amplitude
    float computeCommonMode(std::vector<CommonMode> & summedChargesContainer, std::vector<CommonMode> & commonModeContainer);
      
    /// Get the CRU ID
    /// \return CRU ID
    int getCRU() const {return mCRU;}
      
    /// Get the Time bin
    /// \return Time Bin
    int getTimeBin() const {return mTimeBin;}
      
  private:
    unsigned short      mCRU;           ///< CRU of that Common Mode value
    unsigned short      mTimeBin;       ///< Time bin of that Common Mode value
    float               mCommonMode;    ///< Actual Common Mode value
};
  
}
}

#endif // ALICEO2_TPC_CommonMode_H_
