/// \file DigitTime.h
/// \brief Container class for the Row Digits
/// \author Andi Mathis, andreas.mathis@ph.tum.de
#ifndef ALICEO2_TPC_DigitTime_H_
#define ALICEO2_TPC_DigitTime_H_

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
    DigitTime(int mTimeBin, int nrows);

    /// Destructor
    ~DigitTime();

    /// Resets the container            
    void reset();

    /// Get the size of the container
    /// @return Size of the Row container
    int getSize() {return mRows.size();}

    /// Get the number of entries in the container
    /// @return Number of entries in the Row container
    int getNentries();

    /// Get the time bin
    /// @return time bin          
    int getTimeBin() {return mTimeBin;}

    /// Get the accumulated charge in one time bin
    /// @return Accumulated charge in one time bin
    float getTotalChargeTimeBin() {return mTotalChargeTimeBin;}

    /// Add digit to the row container
    /// @param eventID MC ID of the event
    /// @param trackID MC ID of the track
    /// @param cru CRU of the digit
    /// @param row Pad row of digit
    /// @param pad Pad of digit
    /// @param charge Charge of the digit
    void setDigit(int eventID, int trackID, int cru, int row, int pad, float charge);

    /// Fill output TClonesArray
    /// @param output Output container
    /// @param cru CRU
    /// @param timeBin Time bin
    void fillOutputContainer(TClonesArray *output, int cru, int timeBin);

    /// Fill output TClonesArray
    /// @param output Output container
    /// @param cru CRU
    /// @param timeBin Time bin
    void fillOutputContainer(TClonesArray *output, int cru, int timeBin, std::vector<CommonMode> &commonModeContainer);

  private:
    float                   mTotalChargeTimeBin;        ///< Total accumulated charge in that time bin
    unsigned short          mTimeBin;                   ///< Time bin of that ADC value
    std::vector <DigitRow*> mRows;                      ///< Row Container for the ADC value
};

inline    
void DigitTime::reset()
{  
  for(auto &aRow : mRows) {
    if(aRow == nullptr) continue;
    aRow->reset();
  }
  mTotalChargeTimeBin=0.;
  mRows.clear();
}

inline    
int DigitTime::getNentries()
{
  int counter = 0;
  for(auto &aRow : mRows) {
    if(aRow == nullptr) continue;
    ++ counter;
  }
  return counter;
}

}
}

#endif // ALICEO2_TPC_DigitTime_H_
