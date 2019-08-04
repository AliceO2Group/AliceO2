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

class TBranch;

namespace o2
{
namespace mft
{
class DigitizerTask : public FairTask
{
  using Digitizer = o2::itsmft::Digitizer;

 public:
  DigitizerTask();

  ~DigitizerTask() override;

  InitStatus Init() override;

  void Exec(Option_t* option) override;
  void FinishTask() override;

  Digitizer& getDigitizer() { return mDigitizer; }
  o2::itsmft::DigiParams& getDigiParams() { return mDigitizer.getParams(); }
  const o2::itsmft::DigiParams& getDigiParams() const { return mDigitizer.getParams(); }

  void setContinuous(bool v) { mDigitizer.getParams().setContinuous(v); }
  bool isContinuous() const { return mDigitizer.getParams().isContinuous(); }
  void setFairTimeUnitInNS(double tinNS) { mFairTimeUnitInNS = tinNS < 1. ? 1. : tinNS; }
  double getFairTimeUnitInNS() const { return mFairTimeUnitInNS; }
  void setAlpideROFramLength(float l) { mDigitizer.getParams().setROFrameLength(l); }
  float getAlpideROFramLength() const { return mDigitizer.getParams().getROFrameLength(); }

  void setQEDInput(TBranch* qed, float timebin, UChar_t srcID);

 private:
  void processQEDBackground(double tMax);

  double mFairTimeUnitInNS = 1; ///< Fair time unit in ns
  Int_t mSourceID = 0;          ///< current source
  Int_t mEventID = 0;           ///< current event id from the source
  Digitizer mDigitizer;         ///< Digitizer

  const std::vector<o2::itsmft::Hit>* mHitsArray = nullptr; //! Array of MC hits

  TBranch* mQEDBranch = nullptr;                               //! optional special branch of hits from QED collitions
  const std::vector<o2::itsmft::Hit>* mHitsArrayQED = nullptr; //! array of MC hits from ED
  float mQEDEntryTimeBinNS = 0.f;                              ///< every entry in the QED branch integrates QED for so many nanosec.
  double mLastQEDTimeNS = 0.;                                  ///< center of the time-bin of last added QED bg slot (entry of mQEDBranch)
  int mLastQEDEntry = -1;                                      ///< last used QED entry
  UChar_t mQEDSourceID = 0;                                    ///< MC ID source of the QED (stored in the labels)

  std::vector<o2::itsmft::Digit> mDigitsArray;                     //!  Array of digits
  std::vector<o2::itsmft::Digit>* mDigitsArrayPtr = &mDigitsArray; //! pointer on the digits array

  o2::dataformats::MCTruthContainer<o2::MCCompLabel> mMCTruthArray;                      //! Labels containter
  o2::dataformats::MCTruthContainer<o2::MCCompLabel>* mMCTruthArrayPtr = &mMCTruthArray; //! Labels containter pointer

  ClassDefOverride(DigitizerTask, 1);
};
} // namespace mft
} // namespace o2

#endif
