/// \file DigitADC.h
/// \brief Container class for the ADC values
/// \author Andi Mathis, andreas.mathis@ph.tum.de
#ifndef ALICEO2_TPC_DigitADC_H_
#define ALICEO2_TPC_DigitADC_H_

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
    DigitADC(int eventID, int trackID, float charge);

    /// Destructor
    ~DigitADC();

    /// Get the event ID
    /// @return event ID
    int getMCEventID() {return mEventID;}

    /// Get the track ID
    /// @return track ID
    int getMCTrackID() {return mTrackID;}

    /// Get the ADC value
    /// @return ADC value
    float getADC() {return mADC;}

  private:
    float       mADC;       ///< ADC value of the digit
    int         mEventID;     ///< MC Event ID
    int         mTrackID;     ///< MC Track ID
};
}
}
#endif // ALICEO2_TPC_DigitADC_H_

