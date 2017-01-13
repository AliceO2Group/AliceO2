/// \file Digit.h
/// \brief Digits structure for upgraded TPC
#ifndef ALICEO2_TPC_DIGIT_H
#define ALICEO2_TPC_DIGIT_H

#ifndef __CINT__
#include <boost/serialization/base_object.hpp>  // for base_object
#endif

#include "FairTimeStamp.h"                      // for FairTimeStamp
#include "Rtypes.h"                             // for Float_t, ULong_t, etc
namespace boost { namespace serialization { class access; } }

namespace AliceO2 {
  namespace TPC {

    /// \class Digit
    /// \brief Digit class for the TPC
    ///
    class Digit : public FairTimeStamp {
    public:

      /// Default constructor
      Digit();

      /// Constructor, initializing values for position, charge and time
      /// @param cru CRU of the digit
      /// @param charge Accumulated charge of digit
      /// @param row Row in which the digit was created
      /// @param pad Pad in which the digit was created
      /// @param time Time at which the digit was created
      Digit(Int_t cru, Float_t charge, Int_t row, Int_t pad, Int_t time);

      /// Constructor, initializing values for position, charge, time and common mode
      /// @param cru CRU of the digit
      /// @param charge Accumulated charge of digit
      /// @param row Row in which the digit was created
      /// @param pad Pad in which the digit was created
      /// @param time Time at which the digit was created
      /// @param commonMode Common mode signal on that ROC in the time bin of the digit
      Digit(Int_t cru, Float_t charge, Int_t row, Int_t pad, Int_t time, Float_t commonMode);

      /// Destructor
      virtual ~Digit();

      /// Get the accumulated charged of the digit
      /// @return charge of the digit
      Int_t getCharge() const { return int(mCharge); }

      // Get the accumulated charged of the digit as a float
      /// @return charge of the digit as a float
      Int_t getChargeFloat() const { return mCharge; }

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
      
      UShort_t                  mCRU;
      Float_t                   mCharge;
      UChar_t                   mRow;
      UChar_t                   mPad;
      Float_t                   mCommonMode;
      
      ClassDef(Digit, 3);
    };
  }
}

#endif /* ALICEO2_TPC_DIGIT_H */
