/// \file DigitContainer.h
/// \brief Container class for the CRU Digits
/// \author Andi Mathis, andreas.mathis@ph.tum.de
#ifndef ALICEO2_TPC_DigitContainer_H_
#define ALICEO2_TPC_DigitContainer_H_

#include "TPCSimulation/Digit.h"
#include "TPCSimulation/DigitCRU.h"
#include "TPCSimulation/CommonMode.h"
#include "TPCSimulation/Constants.h"
#include "Rtypes.h"

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
    Int_t getSize() {return mCRU.size();}

    /// Get the number of entires in the container
    /// @return Number of entries in the CRU container
    Int_t getNentries();

    /// Add digit to the container
    /// @param cru CRU of the digit
    /// @param row Pad row of digit
    /// @param pad Pad of digit
    /// @param timeBin Time bin of the digit
    /// @param charge Charge of the digit
    void addDigit(Int_t cru, Int_t timeBin, Int_t row, Int_t pad, Float_t charge);

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
    UShort_t mNCRU;                     ///< CRU of the ADC value
    std::vector<DigitCRU*> mCRU;        ///< CRU Container for the ADC value
};

inline
void DigitContainer::reset() 
{
  for(auto &aCRU : mCRU) {
    if(aCRU == nullptr) continue;
    aCRU->reset();
  }
//       mCRU.clear();
}

inline
Int_t DigitContainer::getNentries() 
{
  Int_t counter = 0;
  for(auto &aCRU : mCRU) {
    if(aCRU == nullptr) continue;
    ++counter;
  }
  return counter;
}

}
}

#endif // ALICEO2_TPC_DigitContainer_H_
