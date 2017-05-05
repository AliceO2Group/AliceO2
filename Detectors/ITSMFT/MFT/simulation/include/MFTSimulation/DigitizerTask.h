/// \file DigitizerTask.h
/// \brief Task driving the conversion from points to digits
/// \author bogdan.vulpescu@cern.ch 
/// \date 03/05/2017

#ifndef ALICEO2_MFT_DIGITIZERTASK_H_
#define ALICEO2_MFT_DIGITIZERTASK_H_

#include "FairTask.h"

#include "MFTSimulation/Digitizer.h"

class TClonesArray;

namespace o2 
{
  namespace MFT 
  {
    class EventHeader; 
    class DigitizerTask : public FairTask
    {
      
    public:
      
      DigitizerTask(Bool_t useAlpide = kFALSE);
      ~DigitizerTask() override;
      
      InitStatus Init() override;
      
      void Exec(Option_t* option) override;
      
      Digitizer& getDigitizer() { return mDigitizer; }
      
    private:
      
      Bool_t mUseAlpideSim; ///< ALPIDE simulation activation flag
      Digitizer mDigitizer; ///< Digitizer
      
      TClonesArray* mPointsArray; ///< Array of MC hits
      TClonesArray* mDigitsArray; ///< Array of digits
      
      ClassDefOverride(DigitizerTask, 1)
	
    };    
  }
}

#endif
