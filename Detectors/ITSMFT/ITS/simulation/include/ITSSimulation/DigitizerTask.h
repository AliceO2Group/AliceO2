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

#include "ITSMFTSimulation/DigiParams.h"
#include "ITSMFTSimulation/Digitizer.h"
#include "ITSMFTSimulation/Hit.h"

namespace o2
{
namespace ITS
{
class DigitizerTask : public FairTask
{
  using Digitizer = o2::ITSMFT::Digitizer;

 public:

  DigitizerTask(Bool_t useAlpide = kFALSE);

  ~DigitizerTask() override;

  InitStatus Init() override;

  void Exec(Option_t* option) override;
  void FinishTask() override;

  Digitizer& getDigitizer() { return mDigitizer; }
  void setContinuous(bool v) { mContinuous = v; }
  bool isContinuous() const { return mContinuous; }
  void setUseAlpideSim(bool v) { mUseAlpideSim = v; }
  bool getUseAlpideSim() const { return mUseAlpideSim; }
  void setFairTimeUnitInNS(double tinNS) { mFairTimeUnitInNS = tinNS < 1. ? 1. : tinNS; }
  double getFairTimeUnitInNS() const { return mFairTimeUnitInNS; }
  void setAlpideROFramLength(float l) { mAlpideROFramLength = l; }
  float getAlpideROFramLength() const { return mAlpideROFramLength; }

 private:

  Bool_t mUseAlpideSim;         ///< ALPIDE simulation activation flag
  Bool_t mContinuous = kFALSE;  ///< flag to do continuous simulation
  double mFairTimeUnitInNS = 1; ///< Fair time unit in ns
  float mAlpideROFramLength = 10000.; ///< ALPIDE ROFrame in ns

  Int_t mSourceID = 0;                  ///< current source
  Int_t mEventID = 0;                   ///< current event id from the source
  Digitizer mDigitizer;                 ///< Digitizer
  const std::vector<o2::ITSMFT::Hit>* mHitsArray = nullptr;   ///< Array of MC hits
  std::vector<o2::ITSMFT::Digit> *mDigitsArray = nullptr; ///< Array of digits

  ClassDefOverride(DigitizerTask, 1);
};
}
}

#endif /* ALICEO2_ITS_DIGITIZERTASK_H */
