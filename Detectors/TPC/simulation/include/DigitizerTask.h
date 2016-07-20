/// \file DigitizerTask.h
/// \brief Task for ALICE TPC digitization
#ifndef __ALICEO2__TPC__DigitizerTask__
#define __ALICEO2__TPC__DigitizerTask__

#include <stdio.h>
#include "FairTask.h"
#include "Rtypes.h"
class TClonesArray;
namespace AliceO2 { namespace TPC { class Digitizer; } }

namespace AliceO2 {
    namespace TPC{

        class Digitizer;

        class DigitizerTask : public FairTask{
        public:
            DigitizerTask();
            virtual ~DigitizerTask();

            virtual InitStatus Init();
            virtual void Exec(Option_t *option);

        private:
            Digitizer           *mDigitizer;

            TClonesArray        *mPointsArray;
            TClonesArray        *mDigitsArray;
            
            ClassDef(AliceO2::TPC::DigitizerTask, 1)
        };
    }
}

#endif /* defined(__ALICEO2__DigitizerTask__) */
