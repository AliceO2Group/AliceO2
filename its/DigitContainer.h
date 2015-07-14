//
//  DigitContainer.h
//  ALICEO2
//
//  Created by Markus Fasel on 25.03.15.
//
//

#ifndef _ALICEO2_ITS_DigitContainer_
#define _ALICEO2_ITS_DigitContainer_

#include "UpgradeGeometryTGeo.h"

namespace AliceO2 {
    namespace ITS{
        
        class Digit;
        class DigitLayer;
        
        class DigitContainer{
        public:
            DigitContainer();
            ~DigitContainer();
            
            void Reset();
            
            void AddDigit(Digit *digi);
            Digit * FindDigit(int layer, int stave, int index);
        
        private:
            DigitLayer                      *fDigitLayer[7];
            UpgradeGeometryTGeo             fGeometry;
        };
    }
}

#endif /* defined(__ALICEO2__DigitContainer__) */
