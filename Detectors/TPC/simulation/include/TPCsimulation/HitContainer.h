/// \file HitContainer.h
/// \brief Container class for the CRU Hits
#ifndef _ALICEO2_HitContainer_
#define _ALICEO2_HitContainer_

#include "TPCsimulation/HitCRU.h"
#include "TPCsimulation/PadHit.h"
#include "Rtypes.h"
#include <map>

class TClonesArray;

namespace AliceO2 {
    namespace TPC{
        class HitPad;
        class HitRow;
        class HitCRU;
        
        /// \class HitContainer
        /// \brief Container class for the individual hits
        
        class HitContainer{
        public:
          
            /// Default constructor
            HitContainer();
            
            /// Destructor
            ~HitContainer();
            
            /// Resets the container
            void reset();
            
            /// Add hit to the CRU container
            /// @param cru CRU of the hit
            /// @param row Row of the hit
            /// @param pad Pad of the hit
            /// @param time Time bin of the hit
            /// @param charge Charge of the hit
            void addHit(Int_t cru, Int_t row, Int_t pad, Int_t time, Float_t charge);
            
            /// Sort hits in pad plane vector
            /// @param padHits Output container
            void getHits(std::vector < AliceO2::TPC::PadHit* > &padHits);
            
        private:
          Int_t mNCRU;
          std::vector<HitCRU*> mCRU;
        };
    }
}

#endif
