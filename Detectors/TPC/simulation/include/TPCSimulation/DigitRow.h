/// \file DigitRow.h
/// \brief Container class for the Pad Digits
/// \author Andi Mathis, andreas.mathis@ph.tum.de
#ifndef ALICEO2_TPC_DigitRow_H_
#define ALICEO2_TPC_DigitRow_H_

#include "TPCSimulation/DigitPad.h"
#include <memory>
#include <iostream>

class TClonesArray;

namespace AliceO2 {
namespace TPC {

/// \class DigitRow
/// \brief Digit container class for the pad digits    

class DigitRow{
  public:

    /// Constructor
    /// @param mRow Row ID
    /// @param npads Number of pads in the row
    DigitRow(int mRow, int npads);

    ///Destructor
    ~DigitRow();

    /// Resets the container
    void reset();

    /// Get the size of the container
    /// @return Size of the pad container
    int getSize() {return mPads.size();}

    /// Get the number of entries in the container
    /// @return Number of entries in the pad container
    int getNentries();

    /// Get the Row ID
    /// @return Row ID
    int getRow() {return mRow;}

    /// Add digit to the pad container
    /// @param pad Pad of the digit
    /// @param charge Charge of the digit
    void setDigit(int eventID, int trackID, int pad, float charge);

    /// Fill output TClonesArray
    /// @param output Output container
    /// @param cru CRU
    /// @param timeBin Time bin
    /// @param row Row
    void fillOutputContainer(TClonesArray *output, int cru, int timeBin, int row);

    /// Fill output TClonesArray
    /// @param output Output container
    /// @param cru CRU
    /// @param timeBin Time bin
    /// @param row Row
    void fillOutputContainer(TClonesArray *output, int cru, int timeBin, int row, float commonMode);

  private:
    unsigned char          mRow;                ///< Row of the ADC value
    std::vector<std::unique_ptr<DigitPad>> mPads;               ///< Pad Container for the ADC value
  
};

inline
void DigitRow::reset()
{
  for(auto &aPad : mPads) {
    if(aPad == nullptr) continue;
    aPad->reset();
  }
  mPads.clear();
}

inline
int DigitRow::getNentries()
{
  int counter = 0;
  for(auto &aPad : mPads) {
    if(aPad == nullptr) continue;
    ++counter;
  }
  return counter;
}

 
}
}

#endif //ALICEO2_TPC_DigitRow_H_
