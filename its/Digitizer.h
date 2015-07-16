/// \file Digitizer.h
/// \brief Task for ALICE ITS digitization
#ifndef ALICEO2_ITS_Digitizer_H_
#define ALICEO2_ITS_Digitizer_H_

#include "TObject.h"
#include "Rtypes.h"

class TClonesArray;

namespace AliceO2{
namespace ITS {
    class DigitContainer;
    class UpgradeGeometryTGeo;
    
    class Digitizer : public TObject {
    public:
        Digitizer();
        ~Digitizer();
            
        void Init();
        DigitContainer *Process(TClonesArray *points);
        
        void SetGainFactor(Double_t gain) { fGain = gain; }
            
    private:
        Digitizer(const Digitizer &);
        Digitizer &operator=(const Digitizer &);
            
        DigitContainer          *fDigitContainer;           ///< Internal digit storage
        UpgradeGeometryTGeo     *fGeometry;                 ///< ITS upgrade geometry
        
        Double_t                fGain;                      ///< pad gain factor (global for the moment)
        
        ClassDef(Digitizer, 1);
    };
}
}

#endif /* ALICEO2_ITS_Digitizer_H_ */
