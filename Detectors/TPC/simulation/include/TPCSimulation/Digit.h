/// \file Digit.h
/// \brief Digits object
/// \author Andi Mathis, andreas.mathis@ph.tum.de
#ifndef ALICEO2_TPC_Digit_H_
#define ALICEO2_TPC_Digit_H_

#ifndef __CINT__
#include <boost/serialization/base_object.hpp>
#endif

#include "FairTimeStamp.h"
#include "Rtypes.h"
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
    Digit(Int_t eventID, Int_t trackID, Int_t cru, Float_t charge, Int_t row, Int_t pad, Int_t time);

    /// Constructor, initializing values for position, charge, time and common mode
    /// @param eventID MC ID of the event
    /// @param trackID MC ID of the track
    /// @param cru CRU of the digit
    /// @param charge Accumulated charge of digit
    /// @param row Row in which the digit was created
    /// @param pad Pad in which the digit was created
    /// @param time Time at which the digit was created
    /// @param commonMode Common mode signal on that ROC in the time bin of the digit
    Digit(Int_t eventID, Int_t trackID, Int_t cru, Float_t charge, Int_t row, Int_t pad, Int_t time, Float_t commonMode);

    /// Destructor
    virtual ~Digit();

    /// Get the event ID
    /// @return event ID
    Int_t getMCEventID() {return mMCEventID;}

    /// Get the track ID
    /// @return track ID
    Int_t getMCTrackID() {return mMCTrackID;}

    /// Get the accumulated charged of the digit
    /// @return charge of the digit
    Int_t getCharge() const { return int(mCharge); }

    /// Get the accumulated charged of the digit as a float
    /// @return charge of the digit as a float
    Float_t getChargeFloat() const { return mCharge; }

    /// Get the CRU of the digit
    /// @return CRU of the digit
    Int_t getCRU() const { return mCRU; }

    /// Get the pad row of the digit
    /// @return pad row of the digit
    Int_t getRow() const { return mRow; }

    /// Get the pad of the digit
    /// @return pad of the digit
    Int_t getPad() const { return mPad; }

    /// Get the timeBin of the digit
    /// @return timeBin of the digit
    Int_t getTimeStamp() const { return int(FairTimeStamp::GetTimeStamp()); }

    /// Get the common mode signal of the digit
    /// @return common mode signal of the digit
    Float_t getCommonMode() const { return mCommonMode; }

    /// Print function: Print basic digit information on the  output stream
    /// @param output Stream to put the digit on
    /// @return The output stream
    std::ostream &Print(std::ostream &output) const;

  private:
    #ifndef __CINT__
    friend class boost::serialization::access;
    #endif
      
    Float_t                   mCharge;          ///< ADC value of the digit
    Float_t                   mCommonMode;      ///< Common mode value of the digit
    Int_t                     mMCEventID;       ///< MC Event ID;
    Int_t                     mMCTrackID;       ///< MC Track ID;
    UShort_t                  mCRU;             ///< CRU of the digit
    UChar_t                   mRow;             ///< Row of the digit
    UChar_t                   mPad;             ///< Pad of the digit
      
  ClassDef(Digit, 3);
};
}
}

#endif // ALICEO2_TPC_Digit_H_
