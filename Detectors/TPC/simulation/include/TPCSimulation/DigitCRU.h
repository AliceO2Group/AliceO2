/// \file DigitCRU.h
/// \brief Digit container for the Time bin Digits
/// \author Andi Mathis, andreas.mathis@ph.tum.de
#ifndef ALICEO2_TPC_DigitCRU_H_
#define ALICEO2_TPC_DigitCRU_H_

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
    DigitCRU(int mCRU);

    /// Destructor
    ~DigitCRU();

    /// Resets the container
    void reset();

    /// Get the number of entries in the container
    /// @return Number of entries in the time bin container
    int getNentries();

    /// Get the size of the container
    /// @return Size of the time bin container
    int getSize() {return mTimeBins.size();}

    /// Get the CRU ID
    /// @return CRU ID
    int getCRUID() {return mCRU;}

    /// Add digit to the row container
    /// @param eventID MC ID of the event
    /// @param trackID MC ID of the track
    /// @param timeBin Time bin of the digit
    /// @param row Pad row of digit
    /// @param pad Pad of digit
    /// @param charge Charge of the digit
    void setDigit(int eventID, int trackID, int timeBin, int row, int pad, float charge);

    /// Fill output TClonesArray
    /// @param output Output container
    /// @param cruID CRU ID
    void fillOutputContainer(TClonesArray *output, int cru);

    /// Fill output TClonesArray
    /// @param output Output container
    /// @param cruID CRU ID
    void fillOutputContainer(TClonesArray *output, int cru, std::vector<CommonMode> &commonModeContainer);

    /// Process Common Mode Information
    /// @param output Output container
    /// @param cruID CRU ID
    void processCommonMode(std::vector<CommonMode> &, int cru);

  private:
    int                    mNTimeBins;        ///< Maximal number of time bins in that CRU
    unsigned short         mCRU;              ///< CRU of the ADC value
    std::vector<std::unique_ptr<DigitTime>> mTimeBins;         ///< Time bin Container for the ADC value
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
int DigitCRU::getNentries()
{
  int counter = 0;
  for(auto &aTime : mTimeBins) {
    if(aTime == nullptr) continue;
    ++counter;
  }
  return counter;
}

}
}

#endif // ALICEO2_TPC_DigitCRU_H_
