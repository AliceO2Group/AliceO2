/// \file DigitizerTask.h
/// \brief Task for ALICE TPC digitization
/// \author Andi Mathis, andreas.mathis@ph.tum.de
#ifndef ALICEO2_TPC_DigitizerTask_H_
#define ALICEO2_TPC_DigitizerTask_H_

#include <stdio.h>
#include <string>
#include "FairTask.h"
#include "FairLogger.h"
#include "TPCSimulation/Digitizer.h"
#include <TClonesArray.h>

namespace AliceO2 {
namespace TPC { 
class Digitizer;
} 
}

namespace AliceO2 {
namespace TPC {
    
// class Digitizer;
    
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

    void setHitFileName(std::string name) { mHitFileName = name; }

    /// Sets the debug flags for the sub-tasks
    /// @param debugsString String containing the debug flags
    ///        o PRFdebug - Debug output after application of the PRF
    void setDebugOutput(TString debugString);
      
    /// Digitization
    /// @param option Option
    virtual void Exec(Option_t *option);
      
  private:
    void fillHitArrayFromFile();

    Digitizer           *mDigitizer;    ///< Digitization process
      
    TClonesArray        *mPointsArray;  ///< Array of detector hits, passed to the digitization
    TClonesArray        *mDigitsArray;  ///< Array of the Digits, passed from the digitization
    
    std::string         mHitFileName;  ///< External hit file exported from AliRoot

  ClassDef(DigitizerTask, 1);
};

inline
void DigitizerTask::setDebugOutput(TString debugString)
{
  LOG(INFO) << "TPC - Debug output enabled for: ";
  if (debugString.Contains("PRFdebug")) {
    LOG(INFO) << "Pad response function, ";
    AliceO2::TPC::Digitizer::setPRFDebug();
  }
  LOG(INFO) << "\n";
}
  
}
}

#endif // ALICEO2_TPC_DigitizerTask_H_
