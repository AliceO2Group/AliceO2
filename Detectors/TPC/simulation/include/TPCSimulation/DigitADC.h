/// \file DigitADC.h
/// \brief Container class for the ADC values
/// \author Andi Mathis, andreas.mathis@ph.tum.de
#ifndef ALICEO2_TPC_DigitADC_H_
#define ALICEO2_TPC_DigitADC_H_

#include "Rtypes.h"

namespace AliceO2 {
namespace TPC {
    
/// \class DigitADC
/// \brief Digit container class for the ADC values
    
class DigitADC{
  public:
      
    /// Default constructor
    DigitADC();
      
    /// Constructor 
    /// @param charge Charge
    DigitADC(Int_t eventID, Int_t trackID, Float_t charge);

    /// Destructor
    ~DigitADC();

    /// Get the event ID
    /// @return event ID
    Int_t getMCEventID() {return mEventID;}

    /// Get the track ID
    /// @return track ID
    Int_t getMCTrackID() {return mTrackID;}

    /// Get the ADC value
    /// @return ADC value
    Float_t getADC() {return mADC;}

  private:
    Int_t mEventID;     ///< MC Event ID
    Int_t mTrackID;     ///< MC Track ID
    Float_t mADC;       ///< ADC value of the digit
};
}
}
#endif // ALICEO2_TPC_DigitADC_H_

