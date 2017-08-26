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

#include "ITSSimulation/Digitizer.h"
#include "ITSMFTSimulation/DigiParams.h"

class TClonesArray;
namespace o2
{
  namespace ITS
  {
    class Digitizer;
  }
} // lines 19-19

namespace o2
{
  namespace ITS
  {
    class Digitizer;

    class DigitizerTask : public FairTask
    {
    public:
      DigitizerTask(Bool_t useAlpide = kFALSE);

      ~DigitizerTask() override;

      InitStatus Init() override;

      void Exec(Option_t* option) override;
      void FinishTask() override;

      Digitizer& getDigitizer() { return mDigitizer; }

      void   setContinuous(bool v) {mContinuous = v;}
      bool   isContinuous()  const {return mContinuous;}

      void   setUseAlpideSim(bool v) { mUseAlpideSim = v;}
      bool   getUseAlpideSim() const { return mUseAlpideSim;}

      void   setFairTimeUnitInNS(double tinNS) {mFairTimeUnitInNS = tinNS<1. ? 1.:tinNS; }
      double getFairTimeUnitInNS() const { return mFairTimeUnitInNS; }
      
    private:
      Bool_t mUseAlpideSim; ///< ALPIDE simulation activation flag
      Bool_t mContinuous = kFALSE; ///< flag to do continuous simulation
      double mFairTimeUnitInNS = 1;  ///< Fair time unit in ns
      Digitizer mDigitizer; ///< Digitizer

      TClonesArray* mPointsArray = nullptr; ///< Array of MC hits
      TClonesArray* mDigitsArray = nullptr; ///< Array of digits

      ClassDefOverride(DigitizerTask, 1)
    };
  }
}

#endif /* ALICEO2_ITS_DIGITIZERTASK_H */
