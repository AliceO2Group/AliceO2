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
#include "FairLogger.h"
#include "FairTask.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "TPCBase/Sector.h"
#include "TPCSimulation/Digitizer.h"
#include "TPCSimulation/SpaceCharge.h"
#include "Steer/HitProcessingManager.h"

class TH3;

namespace o2
{
namespace TPC
{

class Digitizer;

/// \class DigitizerTask
/// This task steers the digitization process and takes care of the input and output
/// Furthermore, it allows for switching debug output on/off

class DigitizerTask : public FairTask
{
 public:
  /// Default constructor
  DigitizerTask(int sectorid = -1);

  /// Destructor
  ~DigitizerTask() override;

  /// Inititializes the digitizer and connects input and output container
  InitStatus Init() override;

  /// Inititializes the digitizer and connects input and output container
  InitStatus Init2();

  /// Sets the debug flags for the sub-tasks
  /// \param debugsString String containing the debug flags
  ///        o PRFdebug - Debug output after application of the PRF
  ///        o DigitMCDebug - Debug output for the DigitMC
  void setDebugOutput(TString debugString);

  /// Switch for triggered / continuous readout
  /// \param isContinuous - false for triggered readout, true for continuous readout
  void setContinuousReadout(bool isContinuous);

  /// query if the r/o mode is continuous
  bool isContinuousReadout() const { return mIsContinuousReadout; }

  /// Enable the use of space-charge distortions
  /// \param distortionType select the type of space-charge distortions (constant or realistic)
  /// \param hisInitialSCDensity optional space-charge density histogram to use at the beginning of the simulation
  /// \param nZSlices number of grid points in z, must be (2**N)+1; default size 129
  /// \param nPhiBins number of grid points in phi; default size 180
  /// \param nRBins number of grid points in r, must be (2**N)+1; default size 129
  void enableSCDistortions(SpaceCharge::SCDistortionType distortionType, TH3* hisInitialSCDensity = nullptr, int nZSlices = 65, int nPhiBins = 180, int nRBins = 65);

  /// Set the maximal number of written out time bins
  /// \param nTimeBinsMax Maximal number of time bins to be written out
  void setMaximalTimeBinWriteOut(int i) { mTimeBinMax = i; }

  /// Setter for time-chunk wise processing
  /// \param isTimeChunk Process time-chunk wise
  void setTimeChunkProcessing(bool isTimeChunk) { mProcessTimeChunks = isTimeChunk; }

  /// Setup a sector for processing
  /// \param s Sector to be processed
  void setupSector(int s);

  void setStartTime(double tstart) { mStartTime = tstart; }
  void setEndTime(double tend) { mEndTime = tend; }

  /// Digitization
  /// \param option Option
  void Exec(Option_t* option) override;

  /// Digitization
  /// \param option Option
  void Exec2(Option_t* option);

  void FinishTask() override;

  void FinishTask2();

  void setData(const std::vector<std::vector<o2::TPC::HitGroup>*>* lefthits,
               const std::vector<std::vector<o2::TPC::HitGroup>*>* righthits,
               const std::vector<o2::TPC::TPCHitGroupID>* leftids, const std::vector<o2::TPC::TPCHitGroupID>* rightids,
               const o2::steer::RunContext* context)
  {
    mAllSectorHitsLeft = lefthits;
    mAllSectorHitsRight = righthits;
    mHitIdsLeft = leftids;
    mHitIdsRight = rightids;
    mRunContext = context;
  }

  void setOutputData(std::vector<o2::TPC::Digit>* digitsArray,
                     o2::dataformats::MCTruthContainer<o2::MCCompLabel>* mcTruthArray)
  {
    mDigitsArray = digitsArray;
    mMCTruthArray = mcTruthArray;
  }

  /// Temporary stuff for bunch train simulation
  ///
  /// Initialise the event times using a bunch train structure
  /// \param numberOfEvents number of event times to simulate
  void initBunchTrainStructure(const size_t numberOfEvents);

 private:
  Digitizer* mDigitizer; ///< Digitization process
  DigitContainer* mDigitContainer;

  std::vector<Digit>* mDigitsArray = nullptr; ///< Array of the Digits, passed from the digitization
  dataformats::MCTruthContainer<MCCompLabel>* mMCTruthArray =
    nullptr; ///< Array for MCTruth information associated to digits in mDigitsArrray. Passed from the digitization
  std::vector<DigitMCMetaData>* mDigitsDebugArray =
    nullptr; ///< Array of the Digits, for debugging purposes only, passed from the digitization

  int mTimeBinMax;           ///< Maximum time bin to be written out
  bool mIsContinuousReadout; ///< Switch for continuous readout
  bool mProcessTimeChunks;   ///< Switch for time-chunk wise processing
  bool mDigitDebugOutput;    ///< Switch for the debug output of the DigitMC
  int mHitSector = -1;       ///< which sector to treat

  const std::vector<o2::TPC::HitGroup>* mSectorHitsArrayLeft;
  const std::vector<o2::TPC::HitGroup>* mSectorHitsArrayRight;

  const std::vector<std::vector<o2::TPC::HitGroup>*>* mAllSectorHitsLeft = nullptr;
  const std::vector<std::vector<o2::TPC::HitGroup>*>* mAllSectorHitsRight = nullptr;
  const std::vector<o2::TPC::TPCHitGroupID>* mHitIdsLeft = nullptr;
  const std::vector<o2::TPC::TPCHitGroupID>* mHitIdsRight = nullptr;
  const o2::steer::RunContext* mRunContext = nullptr;
  double mStartTime; // = tstart [ns]
  double mEndTime;   // = tend [ns]

  // Temporary stuff for bunch train structure simulation
  std::vector<float> mEventTimes; ///< Simulated event times in us
  int mCurrentEvent = 0;          ///< Current event

  ClassDefOverride(DigitizerTask, 1);
};

inline void DigitizerTask::setDebugOutput(TString debugString)
{
  LOG(INFO) << "TPC - Debug output enabled for: ";
  if (debugString.Contains("DigitMCDebug")) {
    LOG(INFO) << "DigitMC, ";
    mDigitDebugOutput = true;
  }
  LOG(INFO) << "\n";
}

inline void DigitizerTask::setContinuousReadout(bool isContinuous)
{
  mIsContinuousReadout = isContinuous;
  o2::TPC::Digitizer::setContinuousReadout(isContinuous);
}

inline void DigitizerTask::setupSector(int s)
{
  mDigitContainer->setup(s);
  if (mDigitsArray)
    mDigitsArray->clear();
  if (mMCTruthArray)
    mMCTruthArray->clear();
  if (mDigitsDebugArray)
    mDigitsDebugArray->clear();
  mHitSector = s;
}
} // namespace TPC
} // namespace o2

#endif // ALICEO2_TPC_DigitizerTask_H_
