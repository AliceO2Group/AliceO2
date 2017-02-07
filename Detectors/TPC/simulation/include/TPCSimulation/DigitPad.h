/// \file DigitPad.h
/// \brief Digit container for the Digits
/// \author Andi Mathis, andreas.mathis@ph.tum.de
#ifndef ALICEO2_TPC_DigitPad_H_
#define ALICEO2_TPC_DigitPad_H_

#include "TPCSimulation/DigitADC.h"
#include "TPCSimulation/CommonMode.h"
#include <TClonesArray.h>

namespace AliceO2 {
namespace TPC {

/// \class DigitPad
/// \brief Digit container class for the digits    

class DigitPad{
  public:

    /// Constructor
    /// @param mPad Pad ID
    DigitPad(int mPad);

    /// Destructor
    ~DigitPad();

    /// Resets the container
    void reset();

    /// Get the size of the container
    /// @return Size of the ADC container
    int getSize() {return mADCCounts.size();}

    /// Get the Pad ID
    /// @return Pad ID
    int getPad() {return mPad;}

    /// Get the accumulated charge on that pad
    /// @return Accumulated charge
    float getTotalChargePad() {return mTotalChargePad;}

    /// Add digit to the time bin container
    /// @param eventID MC ID of the event
    /// @param trackID MC ID of the track
    /// @param charge Charge of the digit
    void setDigit(int eventID, int trackID, float charge);

    /// Fill output TClonesArray
    /// @param output Output container
    /// @param cru CRU ID
    /// @param timeBin Time bin
    /// @param row Row ID
    /// @param pad pad ID
    void fillOutputContainer(TClonesArray *output, int cru, int timeBin, int row, int pad);

    /// Fill output TClonesArray
    /// @param output Output container
    /// @param cru CRU ID
    /// @param timeBin Time bin
    /// @param row Row ID
    /// @param pad pad ID
    void fillOutputContainer(TClonesArray *output, int cru, int timeBin, int row, int pad, float commonMode);

    // Process Common Mode Information
    /// @param output Output container
    /// @param cruID CRU ID
    /// @param timeBin TimeBin
    /// @param rowID Row ID
    /// @param pad pad ID
    void processCommonMode(int cru, int timeBin, int row, int pad);

  private:
    float                  mTotalChargePad;   ///< Total accumulated charge on that pad for a given time bin
    unsigned char          mPad;              ///< Pad of the ADC value
    std::vector <DigitADC>   mADCCounts;        ///< Vector with ADC values
};

inline 
void DigitPad::setDigit(int eventID, int trackID, float charge)
{
  DigitADC digitAdc(eventID, trackID, charge);
  mADCCounts.emplace_back(digitAdc);
}

inline
void DigitPad::reset()
{
  mADCCounts.clear();
}
  
}
}

#endif // ALICEO2_TPC_DigitPad_H_
