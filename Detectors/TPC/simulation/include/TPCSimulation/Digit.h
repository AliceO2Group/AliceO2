/// \file Digit.h
/// \brief Digits object
/// \author Andi Mathis, andreas.mathis@ph.tum.de
#ifndef ALICEO2_TPC_Digit_H_
#define ALICEO2_TPC_Digit_H_

#ifndef __CINT__
#include <boost/serialization/base_object.hpp>
#endif

#include "FairTimeStamp.h"

namespace boost {
namespace serialization {
class access; 
}
}

namespace AliceO2 {
namespace TPC {

/// \class Digit
/// \brief Digit class for the TPC

class Digit : public FairTimeStamp {
  public:

    /// Default constructor
    Digit();

    /// Constructor, initializing values for position, charge and time
    /// @param eventID MC ID of the event
    /// @param trackID MC ID of the track
    /// @param cru CRU of the digit
    /// @param charge Accumulated charge of digit
    /// @param row Row in which the digit was created
    /// @param pad Pad in which the digit was created
    /// @param time Time at which the digit was created
    Digit(int eventID, int trackID, int cru, float charge, int row, int pad, int time);

    /// Constructor, initializing values for position, charge, time and common mode
    /// @param eventID MC ID of the event
    /// @param trackID MC ID of the track
    /// @param cru CRU of the digit
    /// @param charge Accumulated charge of digit
    /// @param row Row in which the digit was created
    /// @param pad Pad in which the digit was created
    /// @param time Time at which the digit was created
    /// @param commonMode Common mode signal on that ROC in the time bin of the digit
    Digit(int eventID, int trackID, int cru, float charge, int row, int pad, int time, float commonMode);

    /// Destructor
    virtual ~Digit();

    /// Get the event ID
    /// @return event ID
    int getMCEventID() {return mMCEventID;}

    /// Get the track ID
    /// @return track ID
    int getMCTrackID() {return mMCTrackID;}

    /// Get the accumulated charged of the digit
    /// @return charge of the digit
    int getCharge() const { return int(mCharge); }

    /// Get the accumulated charged of the digit as a float
    /// @return charge of the digit as a float
    float getChargeFloat() const { return mCharge; }

    /// Get the CRU of the digit
    /// @return CRU of the digit
    int getCRU() const { return mCRU; }

    /// Get the pad row of the digit
    /// @return pad row of the digit
    int getRow() const { return mRow; }

    /// Get the pad of the digit
    /// @return pad of the digit
    int getPad() const { return mPad; }

    /// Get the timeBin of the digit
    /// @return timeBin of the digit
    int getTimeStamp() const { return int(FairTimeStamp::GetTimeStamp()); }

    /// Get the common mode signal of the digit
    /// @return common mode signal of the digit
    float getCommonMode() const { return mCommonMode; }

    /// Print function: Print basic digit information on the  output stream
    /// @param output Stream to put the digit on
    /// @return The output stream
    std::ostream &Print(std::ostream &output) const;

  private:
    #ifndef __CINT__
    friend class boost::serialization::access;
    #endif
      
    float                   mCharge;          ///< ADC value of the digit
    float                   mCommonMode;      ///< Common mode value of the digit
    int                     mMCEventID;       ///< MC Event ID;
    int                     mMCTrackID;       ///< MC Track ID;
    unsigned short          mCRU;             ///< CRU of the digit
    unsigned char           mRow;             ///< Row of the digit
    unsigned char           mPad;             ///< Pad of the digit
      
  ClassDef(Digit, 3);
};
}
}

#endif // ALICEO2_TPC_Digit_H_
