/// \file HitCharge.h
/// \brief Container class for the Charge values of the hits
#ifndef _ALICEO2_TPC_HitCharge_
#define _ALICEO2_TPC_HitCharge_

#include "Rtypes.h"
#include <map>

namespace AliceO2 {
    namespace TPC{
                
        /// \class HitCharge
        /// \brief Hit container class for the Charge values
      
        class HitCharge{
        public:
          
            /// Constructor
            /// @param charge Number of electrons per hit
            HitCharge(Float_t charge);
            
            /// Destructor
            ~HitCharge();
            
            /// Get charge
            /// @return Number of electrons per hit
            Float_t getCharge() {return mCharge;}
                        
        private:
            Float_t mCharge;
        };
    }
}

#endif
