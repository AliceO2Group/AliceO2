/// \file HitTime.h
/// \brief Container class for the Charge Hits
#ifndef _ALICEO2_TPC_HitTime_
#define _ALICEO2_TPC_HitTime_

#include "Rtypes.h"
#include "HitPad.h"

class TClonesArray;

namespace AliceO2 {
    namespace TPC {
        
        class HitCharge;
        
        /// \class HitTime
        /// \brief Container for the charge hits
        
        class HitTime{
        public:
            
            /// Constructor
            /// @param mTimeBin Time bin of the hit
            HitTime(Int_t mTimeBin);
            
            /// Destructor
            ~HitTime();
            
            /// Reset the container
            void reset();
            
            /// Get Time bin
            /// @return Time bin
            Int_t getTimeBin() {return mTimeBin;}
            
            
            /// Add hit to the charge container
            /// @param charge Charge of the hit
            void setHit(Float_t charge);
            
            /// Sort hits in pad plane vector
            /// @param padHits Output container
            /// @param cruID CRU ID
            /// @param rowID Row ID
            /// @param padID Pad ID
            /// @param timeBin Time bin
            Double_t getCharge(Int_t cruID, Int_t rowID, Int_t padID, Int_t timeBin);
            
        private:
            Int_t               mTimeBin;
            HitCharge            *hitCharge;
            std::vector <AliceO2::TPC::HitCharge*>  mChargeCounts;
        };
    }
}

#endif
