/// \file DigitCRU.h
/// \brief Digit container for the Time bin Digits
/// \author Andi Mathis, andreas.mathis@ph.tum.de
#ifndef ALICEO2_TPC_DigitCRU_H_
#define ALICEO2_TPC_DigitCRU_H_

#include "Rtypes.h"
#include "DigitTime.h"
#include "CommonMode.h"

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

    /// Get the number of entries in the container
    /// @return Number of entries in the time bin container
    Int_t getNentries();

    /// Get the size of the container
    /// @return Size of the time bin container
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

    /// Fill output TClonesArray
    /// @param output Output container
    /// @param cruID CRU ID
    void fillOutputContainer(TClonesArray *output, Int_t cru, std::vector<CommonMode> commonModeContainer);

    /// Process Common Mode Information
    /// @param output Output container
    /// @param cruID CRU ID
    void processCommonMode(std::vector<CommonMode> &, Int_t cru);

  private:
    UShort_t                 mCRU;              ///< CRU of the ADC value
    Int_t                    mNTimeBins;        ///< Maximal number of time bins in that CRU
    std::vector <DigitTime*> mTimeBins;         ///< Time bin Container for the ADC value
};
    
    
inline 
void DigitCRU::reset()
{
  for(auto &aTime : mTimeBins) {
    if(aTime == nullptr) continue;
    aTime->reset();
  }
  mTimeBins.clear();
}
    
inline 
Int_t DigitCRU::getNentries() 
{
  Int_t counter = 0;
  for(auto &aTime : mTimeBins) {
    if(aTime == nullptr) continue;
    ++counter;
  }
  return counter;
}

}
}

#endif // ALICEO2_TPC_DigitCRU_H_
