/// \file Digitizer.h
/// \brief Task for ALICE ITS digitization
#ifndef ALICEO2_ITS_Digitizer_H_
#define ALICEO2_ITS_Digitizer_H_

#include "FairTask.h"
#include "RTypes.h"

class TClonesArray;

namespace AliceO2{
namespace ITS {
    class DigitContainer;
    class UpgradeGeometryTGeo;
    
    class Digitizer : public FairTask{
    public:
        Digitizer();
        ~Digitizer();
            
        virtual InitStatus Init();
        virtual void Exec(Option_t *option);
            
    private:
        Digitizer(const Digitizer &);
        Digitizer &operator=(const Digitizer &);
            
        TClonesArray            *fPointsArray;              ///< Input point container
        TClonesArray            *fDigitsArray;              ///< Output digit container
        DigitContainer          *fDigitContainer;           ///< Internal digit storage
        UpgradeGeometryTGeo     *fGeometry;                 ///< ITS upgrade geometry
        
        ClassDef(Digitizer, 1);
    };
}
}

#endif /* ALICEO2_ITS_Digitizer_H_ */
