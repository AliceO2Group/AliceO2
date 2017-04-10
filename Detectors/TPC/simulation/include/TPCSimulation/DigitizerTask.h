/// \file Digitizer.h
/// \brief Definition of the ALICE TPC digitizer task
/// \author Andi Mathis, TU München, andreas.mathis@ph.tum.de

#ifndef ALICEO2_TPC_DigitizerTask_H_
#define ALICEO2_TPC_DigitizerTask_H_

#include <cstdio>
#include <string>
#include "FairTask.h"
#include "FairLogger.h"
#include "TPCSimulation/Digitizer.h"
#include <TClonesArray.h>

namespace o2 {
namespace TPC { 

class Digitizer;
    
/// \class DigitizerTask
/// This task steers the digitization process and takes care of the input and output
/// Furthermore, it allows for switching debug output on/off
    
class DigitizerTask : public FairTask{
  public:
      
    /// Default constructor
    DigitizerTask();
      
    /// Destructor
    ~DigitizerTask() override;
      
    /// Inititializes the digitizer and connects input and output container
    InitStatus Init() override;

    void setHitFileName(std::string name) { mHitFileName = name; }

    /// Sets the debug flags for the sub-tasks
    /// \param debugsString String containing the debug flags
    ///        o PRFdebug - Debug output after application of the PRF
    void setDebugOutput(TString debugString);
      
    /// Digitization
    /// \param option Option
    void Exec(Option_t *option) override;
      
    void FinishTask() override;

  private:
    void fillHitArrayFromFile();

    Digitizer           *mDigitizer;    ///< Digitization process
    DigitContainer      *mDigitContainer;
      
    TClonesArray        *mPointsArray;  ///< Array of detector hits, passed to the digitization
    TClonesArray        *mDigitsArray;  ///< Array of the Digits, passed from the digitization
    
    std::string         mHitFileName;  ///< External hit file exported from AliRoot

  ClassDefOverride(DigitizerTask, 1);
};

inline
void DigitizerTask::setDebugOutput(TString debugString)
{
  LOG(INFO) << "TPC - Debug output enabled for: ";
  if (debugString.Contains("PRFdebug")) {
    LOG(INFO) << "Pad response function, ";
    o2::TPC::Digitizer::setPRFDebug();
  }
  LOG(INFO) << "\n";
}
  
}
}

#endif // ALICEO2_TPC_DigitizerTask_H_
