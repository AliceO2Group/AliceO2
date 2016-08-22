/// \file Digit.h
/// \brief Digits structure for upgraded TPC
#ifndef ALICEO2_TPC_DIGIT_H
#define ALICEO2_TPC_DIGIT_H

#ifndef __CINT__
#include <boost/serialization/base_object.hpp>  // for base_object
#endif

#include "FairTimeStamp.h"                      // for FairTimeStamp
#include "Rtypes.h"                             // for Double_t, ULong_t, etc
namespace boost { namespace serialization { class access; } }

namespace AliceO2{
  namespace TPC{
    
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
      /// @param timestamp Time at which the digit was created
      Digit(Int_t cru, Double_t charge, Int_t row, Int_t pad, Double_t time);
      
      /// Destructor
      virtual ~Digit();
      
      /// Get the accumulated charged of the digit
      /// @return charge of the digit
      Double_t getCharge() const { return mCharge; }
      
      /// Get the CRU of the digit
      /// @return CRU of the digit
      Int_t getCRU() const { return mCRU; }
      
      /// Get the pad row of the digit
      /// @return pad row of the digit
      Int_t getRow() const { return mRow; }
      
      /// Get the pad of the digit
      /// @return pad of the digit
      Int_t getPad() const { return mPad; }
      //       Double_t GetTime() const { return mTimeStamp; }
      Int_t getTimeStamp() const { return Int_t(FairTimeStamp::GetTimeStamp()); }
                  
      /// Set the charge of the digit
      /// @param charge The charge of the the digit
      void setCharge(Double_t charge) { mCharge = charge; }
      
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
            
      ClassDef(Digit, 1);
    };
  }
}

#endif /* ALICEO2_TPC_DIGIT_H */
