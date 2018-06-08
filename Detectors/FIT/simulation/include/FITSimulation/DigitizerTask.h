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

#include "FITBase/Digit.h"
#include "FITSimulation/Detector.h" // for HitType
#include "FITSimulation/Digitizer.h"

namespace o2
{
namespace fit
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

 private:
  Bool_t mContinuous = kFALSE;  ///< flag to do continuous simulation
  double mFairTimeUnitInNS = 1; ///< Fair time unit in ns

  Int_t mSourceID = 0;                                       ///< current source
  Int_t mEventID = 0;                                        ///< current event id from the source
  Digitizer mDigitizer;                                      ///< Digitizer
  const std::vector<o2::fit::HitType>* mHitsArray = nullptr; ///< Array of MC hits

  Digit *mEventDigit = nullptr;       ///< one digit for one event
  //std::vector<o2::fit::Digit>* mDigitsArray = nullptr;       ///< Array of digits

  ClassDefOverride(DigitizerTask, 1);
};
} // namespace fit
} // namespace o2

#endif /* ALICEO2_TOF_DIGITIZERTASK_H */
