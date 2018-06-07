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
/// \brief Task driving the conversion from points to digits (MFT)
/// \author bogdan.vulpescu@cern.ch
/// \date 03/05/2017

#ifndef ALICEO2_MFT_DIGITIZERTASK_H
#define ALICEO2_MFT_DIGITIZERTASK_H

#include <cstdio>
#include <memory>
#include "FairTask.h"
#include "Rtypes.h"

#include "ITSMFTSimulation/DigiParams.h"
#include "ITSMFTSimulation/Digitizer.h"
#include "ITSMFTSimulation/Hit.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "SimulationDataFormat/MCCompLabel.h"

namespace o2
{
namespace MFT
{
class DigitizerTask : public FairTask
{
  using Digitizer = o2::ITSMFT::Digitizer;

 public:
  DigitizerTask();

  ~DigitizerTask() override;

  InitStatus Init() override;

  void Exec(Option_t* option) override;
  void FinishTask() override;

  Digitizer& getDigitizer() { return mDigitizer; }
  o2::ITSMFT::DigiParams& getDigiParams() { return (o2::ITSMFT::DigiParams&)mParams; }

  void setContinuous(bool v) { mParams.setContinuous(v); }
  bool isContinuous() const { return mParams.isContinuous(); }
  void setFairTimeUnitInNS(double tinNS) { mFairTimeUnitInNS = tinNS < 1. ? 1. : tinNS; }
  double getFairTimeUnitInNS() const { return mFairTimeUnitInNS; }
  void setAlpideROFramLength(float l) { mParams.setROFrameLenght(l); }
  float getAlpideROFramLength() const { return mParams.getROFrameLenght(); }

 private:
  o2::ITSMFT::DigiParams mParams; // settings, eventually load this from the CCDB

  double mFairTimeUnitInNS = 1; ///< Fair time unit in ns

  Int_t mSourceID = 0;                                      ///< current source
  Int_t mEventID = 0;                                       ///< current event id from the source
  Digitizer mDigitizer;                                     ///< Digitizer
  const std::vector<o2::ITSMFT::Hit>* mHitsArray = nullptr; //! Array of MC hits

  std::vector<o2::ITSMFT::Digit> mDigitsArray;                     //!  Array of digits
  std::vector<o2::ITSMFT::Digit>* mDigitsArrayPtr = &mDigitsArray; //! pointer on the digits array

  o2::dataformats::MCTruthContainer<o2::MCCompLabel> mMCTruthArray;                      //! Labels containter
  o2::dataformats::MCTruthContainer<o2::MCCompLabel>* mMCTruthArrayPtr = &mMCTruthArray; //! Labels containter pointer

  ClassDefOverride(DigitizerTask, 1);
};
}
}

#endif
