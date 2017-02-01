/// \file DigitRow.h
/// \brief Container class for the Pad Digits
/// \author Andi Mathis, andreas.mathis@ph.tum.de
#ifndef ALICEO2_TPC_DigitRow_H_
#define ALICEO2_TPC_DigitRow_H_

#include "Rtypes.h"
#include "TPCSimulation/DigitPad.h"

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
    DigitRow(Int_t mRow, Int_t npads);

    ///Destructor
    ~DigitRow();

    /// Resets the container
    void reset();

    /// Get the size of the container
    /// @return Size of the pad container
    Int_t getSize() {return mPads.size();}

    /// Get the number of entries in the container
    /// @return Number of entries in the pad container
    Int_t getNentries();

    /// Get the Row ID
    /// @return Row ID
    Int_t getRow() {return mRow;}

    /// Get the accumulated charge in that row
    /// @return Accumulated charge in that row
    Float_t getTotalChargeRow() {return mTotalChargeRow;}

    /// Add digit to the pad container
    /// @param pad Pad of the digit
    /// @param charge Charge of the digit
    void setDigit(Int_t eventID, Int_t trackID, Int_t pad, Float_t charge);

    /// Fill output TClonesArray
    /// @param output Output container
    /// @param cru CRU
    /// @param timeBin Time bin
    /// @param row Row
    void fillOutputContainer(TClonesArray *output, Int_t cru, Int_t timeBin, Int_t row);

    /// Fill output TClonesArray
    /// @param output Output container
    /// @param cru CRU
    /// @param timeBin Time bin
    /// @param row Row
    void fillOutputContainer(TClonesArray *output, Int_t cru, Int_t timeBin, Int_t row, Float_t commonMode);

    /// Process Common Mode Information
    /// @param output Output container
    /// @param cruID CRU ID
    /// @param timeBin TimeBin
    /// @param rowID Row ID
    void processCommonMode(Int_t cru, Int_t timeBin, Int_t row);

  private:
    Float_t                mTotalChargeRow;     ///< Total accumulated charge in that pad row for a given time bin
    UChar_t                mRow;                ///< Row of the ADC value
    std::vector<DigitPad*> mPads;               ///< Pad Container for the ADC value
  
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
Int_t DigitRow::getNentries()
{
  Int_t counter = 0;
  for(auto &aPad : mPads) {
    if(aPad == nullptr) continue;
    ++counter;
  }
  return counter;
}

 
}
}

#endif //ALICEO2_TPC_DigitRow_H_
