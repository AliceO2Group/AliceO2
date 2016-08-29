/// \file PadHitTime.h
/// \brief Time bins contained by one PadHit
#ifndef ALICEO2_TPC_PadHitTime_H
#define ALICEO2_TPC_PadHitTime_H

#include "FairTimeStamp.h"
#include "Rtypes.h"

namespace AliceO2{
  namespace TPC{
    
    class PadHit;
    
    /// \class PadHitTime
    /// \brief Time bin hit objects for the PadHit class
    
    class PadHitTime {
    public:
      
      /// Default constructor
      PadHitTime();
      
      /// Constructor
      /// @param time Time bin of the hit
      /// @param charge Number of electrons of the hit
      PadHitTime(Double_t time, Double_t charge);
      
      /// Destructor
      virtual ~PadHitTime();
      
      /// Reset the object
      void reset();
      
      /// Get the time bin
      /// @return Time bin
      Double_t getTime() const { return mTimeBin; }
      
      /// Get the charge
      /// @return Charge
      Double_t getCharge() const { return mCharge; }
      
    private:
      
      UChar_t                   mTimeBin;
      Float_t                   mCharge;
      
      ClassDef(PadHitTime, 1);
    };
  }
}

#endif /* ALICEO2_TPC_PadHitTime_H */
