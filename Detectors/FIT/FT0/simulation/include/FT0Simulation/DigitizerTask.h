// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file DigitizerTask.h
/// \brief Definition of the TOF digitizer task

#ifndef ALICEO2_FIT_DIGITIZERTASK_H
#define ALICEO2_FIT_DIGITIZERTASK_H

#include "FairTask.h" // for FairTask, InitStatus
#include "Rtypes.h"   // for DigitizerTask::Class, ClassDef, etc

#include "DataFormatsFT0/Digit.h"
#include "FT0Simulation/Detector.h" // for HitType
#include "FITSimulation/Digitizer.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "DataFormatsFT0/MCLabel.h"

namespace o2
{
namespace ft0
{

class DigitizerTask : public FairTask
{
  using Digitizer = o2::fit::Digitizer;

 public:
  DigitizerTask();
  ~DigitizerTask() override;

  InitStatus Init() override;

  void Exec(Option_t* option) override;
  void FinishTask() override;

  Digitizer& getDigitizer() { return mDigitizer; }
  void setContinuous(bool v) { mContinuous = v; }
  bool isContinuous() const { return mContinuous; }

  void setQEDInput(TBranch* qed, float timebin, UChar_t srcID);

 private:
  void processQEDBackground(double tMax);

  Bool_t mContinuous = kFALSE;  ///< flag to do continuous simulation
  double mFairTimeUnitInNS = 1; ///< Fair time unit in ns

  Int_t mSourceID = 0;                                       ///< current source
  Int_t mEventID = 0;                                        ///< current event id from the source
  Digitizer mDigitizer;                                      ///< Digitizer
  const std::vector<o2::ft0::HitType>* mHitsArray = nullptr; ///< Array of MC hits

  TBranch* mQEDBranch = nullptr;                                //! optional special branch of hits from QED collitions
  const std::vector<o2::ft0::HitType>* mHitsArrayQED = nullptr; //! array of MC hits from ED
  float mQEDEntryTimeBinNS = 0.f;                               ///< every entry in the QED branch integrates QED for so many nanosec.
  double mLastQEDTimeNS = 0.;                                   ///< center of the time-bin of last added QED bg slot (entry of mQEDBranch)
  int mLastQEDEntry = -1;                                       ///< last used QED entry
  UChar_t mQEDSourceID = 0;                                     ///< MC ID source of the QED (stored in the labels)

  o2::ft0::Digit* mEventDigit = nullptr;
  o2::dataformats::MCTruthContainer<o2::ft0::MCLabel> mMCTruthArray;                      //! Labels containter
  o2::dataformats::MCTruthContainer<o2::ft0::MCLabel>* mMCTruthArrayPtr = &mMCTruthArray; //! Labels containter pointer

  ClassDefOverride(DigitizerTask, 1);
};
} // namespace ft0
} // namespace o2

#endif /* ALICEO2_TOF_DIGITIZERTASK_H */
