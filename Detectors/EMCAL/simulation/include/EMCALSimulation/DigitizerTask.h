// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef ALICEO2_EMCAL_DIGITIZERTASK_H
#define ALICEO2_EMCAL_DIGITIZERTASK_H

#include <cstdio>
#include "FairTask.h" // for FairTask, InitStatus
#include "Rtypes.h"   // for DigitizerTask::Class, ClassDef, etc

#include "EMCALBase/Hit.h"
#include "EMCALSimulation/Digitizer.h"

namespace o2
{
namespace emcal
{

/// \class DigitizerTask
/// \brief FairTask running EMCAL digitization
/// \ingroup EMCALsimulation
/// \author Anders Knospe, University of Houston
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
  double mFairTimeUnitInNS = 1;                 ///< Fair time unit in ns
  Int_t mSourceID = 0;                          ///< current source
  Int_t mEventID = 0;                           ///< current event id from the source
  Digitizer mDigitizer;                         ///< Digitizer
  const std::vector<Hit>* mHitsArray = nullptr; ///< Array of MC hits
  std::vector<Digit>* mDigitsArray = nullptr;   ///< Array of digits

  ClassDefOverride(DigitizerTask, 1);
};
} // namespace emcal
} // namespace o2

#endif /* ALICEO2_EMCAL_DIGITIZERTASK_H */
