/// \file HitCRU.h
/// \brief Hit container for the Row Hits
#ifndef _ALICEO2_TPC_HitCRU_
#define _ALICEO2_TPC_HitCRU_

#include "Rtypes.h"
#include "PadHit.h"

class TClonesArray;

namespace AliceO2 {
    namespace TPC {
        
        class HitADC;
        class HitTime;
        class HitPad;
        class HitRow;
        
        /// \class HitCRU
        /// \brief Hit container class for the row hits
        
        class HitCRU{
        public:
            
            /// Constructor
            /// @param mCRUID CRU ID
            /// @param nrows Number of pad rows in the CRU
            HitCRU(Int_t mCRUID, Int_t nrows);
            
            /// Destructor
            ~HitCRU();
            
            /// Resets the row hit container
            void reset();
            
            /// Get the CRU ID
            /// @return CRU ID
            Int_t getCRUID() {return mCRUID;}
            
            /// Add hit to the row container
            /// @param row Row of the hit
            /// @param pad Pad of the hit
            /// @param time Time bin of the hit
            /// @param charge Charge of the hit
            void setHit(Int_t row, Int_t pad, Int_t time, Float_t charge);
            
            /// Sort hits in pad plane vector
            /// @param padHits Output container
            /// @param cruID CRU ID
            void getHits(std::vector < AliceO2::TPC::PadHit* > &padHits, Int_t cruID);
            
        private:
            Int_t               mCRUID;
            Int_t               mNRows;
            std::vector <HitRow*>  mRows;
        };
    }
}

#endif
