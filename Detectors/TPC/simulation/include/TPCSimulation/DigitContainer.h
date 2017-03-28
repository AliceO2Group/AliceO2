/// \file DigitContainer.h
/// \brief Container class for the CRU Digits
/// \author Andi Mathis, andreas.mathis@ph.tum.de
#ifndef ALICEO2_TPC_DigitContainer_H_
#define ALICEO2_TPC_DigitContainer_H_

#include "TPCSimulation/Digit.h"
#include "TPCBase/CRU.h"
#include "TPCSimulation/DigitCRU.h"
#include "TPCSimulation/CommonMode.h"
#include "TPCSimulation/Constants.h"

class TClonesArray;

namespace AliceO2 {
namespace TPC {

/// \class DigitContainer
/// \brief Digit container class

class DigitContainer{
  public:
    
    /// Default constructor
    DigitContainer();

    /// Destructor
    ~DigitContainer();

    void reset();

    /// Get the size of the container
    /// @return Size of the CRU container
    int getSize() {return mCRU.size();}

    /// Get the number of entires in the container
    /// @return Number of entries in the CRU container
    int getNentries();

    /// Add digit to the container
    /// @param eventID MC ID of the event
    /// @param trackID MC ID of the track
    /// @param cru CRU of the digit
    /// @param row Pad row of digit
    /// @param pad Pad of digit
    /// @param timeBin Time bin of the digit
    /// @param charge Charge of the digit
    void addDigit(int eventID, int trackID, int cru, int timeBin, int row, int pad, float charge);

    /// Fill output TClonesArray
    /// @param output Output container
    void fillOutputContainer(TClonesArray *output);

    /// Fill output TClonesArray
    /// @param output Output container
    void fillOutputContainer(TClonesArray *output, std::vector<CommonMode> &commonModeContainer);

    /// Process Common Mode Information
    /// @param output Output container
    void processCommonMode(std::vector<CommonMode> &);

  private:
    std::array<std::unique_ptr<DigitCRU> , CRU::MaxCRU> mCRU;   ///< CRU Container for the ADC value
};

inline
void DigitContainer::reset() 
{
  for(auto &aCRU : mCRU) {
    if(aCRU == nullptr) continue;
    aCRU->reset();
  }
  std::fill(mCRU.begin(),mCRU.end(), nullptr);
}

inline
int DigitContainer::getNentries()
{
  int counter = 0;
  for(auto &aCRU : mCRU) {
    if(aCRU == nullptr) continue;
    ++counter;
  }
  return counter;
}

}
}

#endif // ALICEO2_TPC_DigitContainer_H_
