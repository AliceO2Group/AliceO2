/// \file DigitizerTask.h
/// \brief Task for ALICE TPC digitization
#ifndef __ALICEO2_TPC_DigitizerTask__
#define __ALICEO2_TPC_DigitizerTask__

#include <stdio.h>
#include "FairTask.h"
#include "Rtypes.h"
class TClonesArray;
namespace AliceO2 { namespace TPC { class Digitizer; } }

namespace AliceO2 {
  namespace TPC {
    
    class Digitizer;
    
    /// \class DigitizerTask
    /// \brief Digitizer task for the TPC
    
    class DigitizerTask : public FairTask{
    public:
      
      /// Default constructor
      DigitizerTask();
      
      /// Destructor
      virtual ~DigitizerTask();
      
      /// Inititializes the digitizer and connects input and output container
      virtual InitStatus Init();
      
      /// Digitization
      /// @param option Option
      virtual void Exec(Option_t *option);
      
    private:
      Digitizer           *mDigitizer;
      
      TClonesArray        *mPointsArray;
      TClonesArray        *mDigitsArray;
      
      ClassDef(DigitizerTask, 1)
    };
  }
}

#endif /* defined(__ALICEO2__DigitizerTask__) */
