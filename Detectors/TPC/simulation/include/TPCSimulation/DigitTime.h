/// \file DigitTime.h
/// \brief Container class for the Row Digits
/// \author Andi Mathis, andreas.mathis@ph.tum.de
#ifndef ALICEO2_TPC_DigitTime_H_
#define ALICEO2_TPC_DigitTime_H_

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
    /// @param eventID MC ID of the event
    /// @param trackID MC ID of the track
    /// @param cru CRU of the digit
    /// @param row Pad row of digit
    /// @param pad Pad of digit
    /// @param charge Charge of the digit
    void setDigit(Int_t eventID, Int_t trackID, Int_t cru, Int_t row, Int_t pad, Float_t charge);

    /// Fill output TClonesArray
    /// @param output Output container
    /// @param cru CRU
    /// @param timeBin Time bin
    void fillOutputContainer(TClonesArray *output, Int_t cru, Int_t timeBin);

    /// Fill output TClonesArray
    /// @param output Output container
    /// @param cru CRU
    /// @param timeBin Time bin
    void fillOutputContainer(TClonesArray *output, Int_t cru, Int_t timeBin, std::vector<CommonMode> &commonModeContainer);

    /// Process Common Mode Information
    /// @param output Output container
    /// @param cru CRU
    /// @param timeBin Time bin
    void processCommonMode(Int_t cru, Int_t timeBin);

  private:
    UShort_t                mTimeBin;                   ///< Time bin of that ADC value
    UChar_t                 mNRows;                     ///< Number of pad rows in that CRU for a given time bin
    Float_t                 mTotalChargeTimeBin;        ///< Total accumulated charge in that time bin
    std::vector <DigitRow*> mRows;                      ///< Row Container for the ADC value
};

inline    
void DigitTime::reset()
{  
  for(auto &aRow : mRows) {
    if(aRow == nullptr) continue;
    aRow->reset();
  }
  mRows.clear();
}

inline    
Int_t DigitTime::getNentries() 
{
  Int_t counter = 0;
  for(auto &aRow : mRows) {
    if(aRow == nullptr) continue;
    ++ counter;
  }
  return counter;
}

}
}

#endif // ALICEO2_TPC_DigitTime_H_
