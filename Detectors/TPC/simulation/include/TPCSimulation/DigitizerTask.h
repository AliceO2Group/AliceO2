// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

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
#include "TPCBase/Sector.h"
#include "SimulationDataFormat/MCTruthContainer.h"

namespace o2 {
namespace TPC { 

class Digitizer;
    
/// \class DigitizerTask
/// This task steers the digitization process and takes care of the input and output
/// Furthermore, it allows for switching debug output on/off
    
class DigitizerTask : public FairTask{
  public:
      
    /// Default constructor
    DigitizerTask(int sectorid=-1);
      
    /// Destructor
    ~DigitizerTask() override;
      
    /// Inititializes the digitizer and connects input and output container
    InitStatus Init() override;

    /// Sets the debug flags for the sub-tasks
    /// \param debugsString String containing the debug flags
    ///        o PRFdebug - Debug output after application of the PRF
    ///        o DigitMCDebug - Debug output for the DigitMC
    void setDebugOutput(TString debugString);

    /// Switch for triggered / continuous readout
    /// \param isContinuous - false for triggered readout, true for continuous readout
    void setContinuousReadout(bool isContinuous);

    /// Set the maximal number of written out time bins
    /// \param nTimeBinsMax Maximal number of time bins to be written out
    void setMaximalTimeBinWriteOut(int i) { mTimeBinMax = i; }
      
    /// Digitization
    /// \param option Option
    void Exec(Option_t *option) override;
      
    void FinishTask() override;

    /// Temporary stuff for bunch train simulation
    ///
    /// Initialise the event times using a bunch train structure
    /// \param numberOfEvents number of event times to simulate
    void initBunchTrainStructure(const size_t numberOfEvents);
  private:
    Digitizer           *mDigitizer;    ///< Digitization process
    DigitContainer      *mDigitContainer;
      
    std::vector<o2::TPC::Digit> *mDigitsArray = nullptr;  ///< Array of the Digits, passed from the digitization
    o2::dataformats::MCTruthContainer<o2::MCCompLabel> *mMCTruthArray = nullptr; ///< Array for MCTruth information associated to digits in mDigitsArrray. Passed from the digitization
    std::vector<o2::TPC::DigitMCMetaData> *mDigitsDebugArray = nullptr;  ///< Array of the Digits, for debugging purposes only, passed from the digitization
    
    int                 mTimeBinMax;   ///< Maximum time bin to be written out
    bool                mIsContinuousReadout; ///< Switch for continuous readout
    bool                mDigitDebugOutput;    ///< Switch for the debug output of the DigitMC
    int                 mHitSector=-1; ///< which sector to treat

    const std::vector<o2::TPC::HitGroup> *mSectorHitsArray[Sector::MAXSECTOR];

    // Temporary stuff for bunch train structure simulation
    std::vector<float> mEventTimes; ///< Simulated event times in us
    int                mCurrentEvent = 0; ///< Current event

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
  if (debugString.Contains("DigitMCDebug")) {
    LOG(INFO) << "DigitMC, ";
    mDigitDebugOutput = true;
  }
  LOG(INFO) << "\n";
}
  
inline
void DigitizerTask::setContinuousReadout(bool isContinuous)
{
  mIsContinuousReadout = isContinuous;
  o2::TPC::Digitizer::setContinuousReadout(isContinuous);
}

}
}

#endif // ALICEO2_TPC_DigitizerTask_H_
