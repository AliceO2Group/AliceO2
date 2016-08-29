/// \file HitPad.h
/// \brief Hit container for the TimeBin Hits
#ifndef _ALICEO2_TPC_HitPad_
#define _ALICEO2_TPC_HitPad_

#include "Rtypes.h"
#include "TPCsimulation/PadHit.h"

class TClonesArray;

namespace AliceO2 {
    namespace TPC {
        
        class HitADC;
        class HitTime;
        class HitPad;
        
        /// \class HitPad
        /// \brief Container class for the time bin hits
        
        class HitPad{
        public:
            /// Constructor
            /// @param mPadID Pad ID
            HitPad(Int_t mPadID);
            
            /// Destructor
            ~HitPad();
            
            /// Reset the hit container
            void reset();
            
            /// Get the Pad ID
            /// @return Pad ID
            Int_t getPad() {return mPadID;}
            
            /// Add hit to the row container
            /// @param time Time bin of the hit
            /// @param charge Charge of the hit
            void setHit(Int_t time, Float_t charge);
            
            /// Sort hits in pad plane vector
            /// @param padHits Output container
            /// @param cruID CRU ID
            /// @param rowID Row ID
            /// @param padID Pad ID
            void getHits(std::vector < PadHit* > &padHits, Int_t cruID, Int_t rowID, Int_t padID);
            
        private:
            Int_t               mPadID;
            Int_t               mNTimeBins;
            std::vector <HitTime*>  mTimeBins;
        };
    }
}

#endif
