/// \file DigitizerTask.h
/// \brief Definition of the ITS digitizer task

//
//  Created by Markus Fasel on 16.07.15.
//
//

#ifndef ALICEO2_ITS_DIGITIZERTASK_H
#define ALICEO2_ITS_DIGITIZERTASK_H

#include <cstdio>
#include "FairTask.h" // for FairTask, InitStatus
#include "Rtypes.h"   // for DigitizerTask::Class, ClassDef, etc

#include "ITSSimulation/Digitizer.h"

class TClonesArray;
namespace o2
{
  namespace ITS
  {
    class Digitizer;
  }
} // lines 19-19

namespace o2
{
  namespace ITS
  {
    class Digitizer;

    class DigitizerTask : public FairTask
    {
    public:
      DigitizerTask(Bool_t useAlpide = kFALSE);

      virtual ~DigitizerTask();

      virtual InitStatus Init();

      virtual void Exec(Option_t* option);

      Digitizer& getDigitizer() { return mDigitizer; }
    private:
      Bool_t mUseAlpideSim; ///< ALPIDE simulation activation flag
      Digitizer mDigitizer; ///< Digitizer

      TClonesArray* mPointsArray; ///< Array of MC hits
      TClonesArray* mDigitsArray; ///< Array of digits

      ClassDef(DigitizerTask, 1)
    };
  }
}

#endif /* ALICEO2_ITS_DIGITIZERTASK_H */
