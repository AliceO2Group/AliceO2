/// \file PadHit.h
/// \brief Pad hit structure for upgraded TPC
#ifndef ALICEO2_TPC_PadHit_H
#define ALICEO2_TPC_PadHit_H

#include "FairTimeStamp.h"
#include "PadHitTime.h"
#include "Rtypes.h"

namespace AliceO2{
  namespace TPC{
    
    class PadHitTime;
    
    /// \class PadHit
    /// \brief PadHit container class containing all time bin hits on one single pad of the pad plane
    
    class PadHit {
    public:
      
      /// Default constructor
      PadHit();
      
      /// Constructor
      /// @param cru CRU ID
      /// @param row Pad row
      /// @param pad Pad
      PadHit(Int_t cru, Int_t row, Int_t pad);
      
      /// Destructor
      virtual ~PadHit();
      
      /// Add time bin hits (PadHitTime object) to the PadHit
      /// @param time Time of the hit
      /// @param charge Charge of the hit
      void addTimeHit(Double_t time, Double_t charge);
      
      /// Get all time bin hit objects on one single pad
      /// @return Vector with all Time bin hits
      std::vector < AliceO2::TPC::PadHitTime* > getTimeHit() {return mTimeHits; }
      
      /// Get the CRU ID
      /// @return CRU ID
      Int_t getCRU() const { return mCRU; }
      
      /// Get the Row ID
      /// @return Row ID
      Int_t getRow() const { return mRow; }
      
      /// Get the Pad ID
      /// @return Pad ID
      Int_t getPad() const { return mPad; }
      
      /// Get the number of time bin hits
      /// @return Number of time bin hits
      Int_t getHitTimeSize() const {return mTimeHits.size(); }
      
    private:
      UShort_t                  mCRU;
      UChar_t                   mRow;
      UChar_t                   mPad;
      std::vector < PadHitTime* >  mTimeHits;
      
      ClassDef(PadHit, 1);
    };
  }
}

#endif /* ALICEO2_TPC_PadHit_H */
