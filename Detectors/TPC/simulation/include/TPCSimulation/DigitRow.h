/// \file DigitRow.h
/// \brief Definition of the Row container
/// \author Andi Mathis, TU München, andreas.mathis@ph.tum.de

#ifndef ALICEO2_TPC_DigitRow_H_
#define ALICEO2_TPC_DigitRow_H_

#include "TPCSimulation/DigitPad.h"
#include <memory>

class TClonesArray;

namespace o2 {
namespace TPC {

/// \class DigitRow
/// This is the forth class of the intermediate Digit Containers, in which all incoming electrons from the hits are sorted into after amplification
/// The structure assures proper sorting of the Digits when later on written out for further processing.
/// This class holds the individual Pad containers and is contained within the Time Bin Container.

class DigitRow{
  public:

    /// Constructor
    /// \param mRow Row ID
    /// \param npads Number of pads in the row
    DigitRow(int mRow, int npads);

    ///Destructor
    ~DigitRow() = default;

    /// Resets the container
    void reset();

    /// Get the size of the container
    /// \return Size of the pad container
    size_t getSize() const {return mPads.size();}

    /// Get the number of entries in the container
    /// \return Number of entries in the pad container
    int getNentries() const;

    /// Get the Row ID
    /// \return Row ID
    int getRow() const {return mRow;}

    /// Add digit to the pad container
    /// \param pad Pad of the digit
    /// \param charge Charge of the digit
    void setDigit(int eventID, int trackID, int pad, float charge);

    /// Fill output TClonesArray
    /// \param output Output container
    /// \param cru CRU
    /// \param timeBin Time bin
    /// \param row Row
    void fillOutputContainer(TClonesArray *output, int cru, int timeBin, int row);

    /// Fill output TClonesArray
    /// \param output Output container
    /// \param cru CRU
    /// \param timeBin Time bin
    /// \param row Row
    void fillOutputContainer(TClonesArray *output, int cru, int timeBin, int row, float commonMode);

  private:
    unsigned char          mRow;                ///< Row of the ADC value
    std::vector<std::unique_ptr<DigitPad>> mPads;               ///< Pad Container for the ADC value
  
};

inline
DigitRow::DigitRow(int row, int npads)
  : mRow(row)
  , mPads(npads)
{}

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
int DigitRow::getNentries() const
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
