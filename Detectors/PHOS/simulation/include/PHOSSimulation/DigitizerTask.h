// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef ALICEO2_PHOS_DIGITIZERTASK_H
#define ALICEO2_PHOS_DIGITIZERTASK_H

#include "FairTask.h" // for FairTask, InitStatus
#include "Rtypes.h"   // for DigitizerTask::Class, ClassDef, etc

#include "PHOSBase/Hit.h"
#include "PHOSSimulation/Digitizer.h"

namespace o2
{
namespace phos
{
class DigitizerTask : public FairTask
{
 public:
  DigitizerTask();
  ~DigitizerTask() override;

  InitStatus Init() override;

  void Exec(Option_t* option) override;
  void FinishTask() override;

  Digitizer& getDigitizer() { return mDigitizer; }

  void setFairTimeUnitInNS(double tinNS) { mFairTimeUnitInNS = tinNS < 1. ? 1. : tinNS; }
  double getFairTimeUnitInNS() const { return mFairTimeUnitInNS; }

 private:
  double mFairTimeUnitInNS = 1;                                            ///< Fair time unit in ns
  Int_t mSourceID = 0;                                                     ///< current source
  Int_t mEventID = 0;                                                      ///< current event id from the source
  Digitizer mDigitizer;                                                    ///< Digitizer
  const std::vector<Hit>* mHitsArray = nullptr;                            ///< Array of MC hits
  std::vector<Digit>* mDigitsArray = nullptr;                              ///< Array of digits
  o2::dataformats::MCTruthContainer<o2::phos::MCLabel>* mLabels = nullptr; ///< Array of digit labels

  ClassDefOverride(DigitizerTask, 2);
};
} // namespace phos
} // namespace o2

#endif /* ALICEO2_PHOS_DIGITIZERTASK_H */
