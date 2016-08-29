/// \file HitRow.h
/// \brief Container class for the Row Hits
#ifndef _ALICEO2_TPC_HitRow_
#define _ALICEO2_TPC_HitRow_

#include "Rtypes.h"
#include "TPCsimulation/PadHit.h"

class TClonesArray;

namespace AliceO2 {
    namespace TPC {
        
        class HitADC;
        class HitTime;
        class HitPad;
        
        /// \class HitRow
        /// \brief Container class for the Pad hits
        
        class HitRow{
        public:
          
            /// Constructor
            /// @param mRowID Row ID
            /// @param npads Number of pads in the row
            HitRow(Int_t mRowID, Int_t npads);
            
            /// Destructor
            ~HitRow();
            
            /// Reset the container
            void reset();
            
            /// Get the row ID
            /// @return Row ID
            Int_t getRow() {return mRowID;}
            
            
            /// Add hit to the pad container
            /// @param pad Pad of the hit
            /// @param time Time bin of the hit
            /// @param charge Charge of the hit
            void setHit(Int_t pad, Int_t time, Float_t charge);
                        
            /// Sort hits in pad plane vector
            /// @param padHits Output container
            /// @param cruID CRU ID
            /// @param rowID Row ID
            void getHits(std::vector < PadHit* > &padHits, Int_t cruID, Int_t rowID);
            
        private:
            Int_t               mRowID;
            Int_t               mNPads;
            std::vector<HitPad*> mPads;
        };
    }
}

#endif
