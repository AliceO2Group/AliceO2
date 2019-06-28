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
/// \brief Definition of the FDD digitizer task

#ifndef ALICEO2_FDD_DIGITIZERTASK_H
#define ALICEO2_FDD_DIGITIZERTASK_H

#include "FairTask.h" // for FairTask, InitStatus
#include "Rtypes.h"   // for DigitizerTask::Class, ClassDef, etc

#include "DataFormatsFDD/Digit.h"
#include "FDDSimulation/Detector.h"
#include "FDDSimulation/Digitizer.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "DataFormatsFDD/MCLabel.h"

namespace o2
{
namespace fdd
{

class DigitizerTask : public FairTask
{
  using Digitizer = o2::fdd::Digitizer;

 public:
  DigitizerTask();
  ~DigitizerTask() override;

  InitStatus Init() override;

  void Exec(Option_t* option) override;
  void FinishTask() override;

  Digitizer& getDigitizer() { return mDigitizer; }
  void setContinuous(bool v) { mContinuous = v; }
  bool isContinuous() const { return mContinuous; }

 private:
  Bool_t mContinuous = kFALSE;  ///< flag to do continuous simulation
  double mFairTimeUnitInNS = 1; ///< Fair time unit in ns

  Int_t mSourceID = 0;                                   ///< current source
  Int_t mEventID = 0;                                    ///< current event id from the source
  Digitizer mDigitizer;                                  ///< Digitizer
  const std::vector<o2::fdd::Hit>* mHitsArray = nullptr; ///< Array of MC hits

  o2::fdd::Digit* mEventDigit = nullptr;
  o2::dataformats::MCTruthContainer<o2::fdd::MCLabel> mMCTruthArray;                      //! Labels containter
  o2::dataformats::MCTruthContainer<o2::fdd::MCLabel>* mMCTruthArrayPtr = &mMCTruthArray; //! Labels containter pointer

  ClassDefOverride(DigitizerTask, 1);
};
} // namespace fdd
} // namespace o2

#endif
