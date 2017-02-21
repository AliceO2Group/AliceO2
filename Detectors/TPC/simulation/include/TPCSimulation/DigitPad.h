/// \file DigitPad.h
/// \brief Digit container for the Digits
/// \author Andi Mathis, andreas.mathis@ph.tum.de
#ifndef ALICEO2_TPC_DigitPad_H_
#define ALICEO2_TPC_DigitPad_H_

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

    /// Get the Pad ID
    /// @return Pad ID
    int getPad() {return mPad;}

    /// Get the accumulated charge on that pad
    /// @return Accumulated charge
    float getChargePad() {return mChargePad;}

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

    void processMClabels(std::vector<long> &sortedMCLabels);
    
  private:
    float                  mChargePad;   ///< Total accumulated charge on that pad for a given time bin
    unsigned char          mPad;         ///< Pad of the ADC value
    std::vector<long>      mMCID;        ///< vector containing the MC ID encoded as described below
};

inline 
void DigitPad::setDigit(int eventID, int trackID, float charge)
{
  // the MC ID is encoded such that we can have 999,999 tracks
  // numbers larger than 1000000 correspond to the event ID
  // i.e. 12000010 corresponds to event 12 with track ID 10
  mMCID.emplace_back((eventID)*1000000.f + trackID);
  mChargePad += charge;
}

inline
void DigitPad::reset()
{
  mChargePad = 0;
  mMCID.resize(0);
}
  
}
}

#endif // ALICEO2_TPC_DigitPad_H_
