// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See https://alice-o2.web.cern.ch/ for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file DigitizerTask.h
/// \brief Task driving the conversion from points to digits
/// \author bogdan.vulpescu@cern.ch 
/// \date 03/05/2017

#ifndef ALICEO2_MFT_DIGITIZERTASK_H_
#define ALICEO2_MFT_DIGITIZERTASK_H_

#include "FairTask.h"

#include "TClonesArray.h"

#include "MFTSimulation/Digitizer.h"

namespace o2 
{
  namespace MFT 
  {
    class EventHeader; 
    class DigitizerTask : public FairTask
    {
      
    public:
      
      DigitizerTask(Bool_t useAlpide = kFALSE);
      ~DigitizerTask() override;
      
      InitStatus Init() override;
      
      void Exec(Option_t* option) override;
      
      Digitizer& getDigitizer() { return mDigitizer; }
      
    private:
      
      Bool_t mUseAlpideSim; ///< ALPIDE simulation activation flag
      Digitizer mDigitizer; ///< Digitizer
      
      TClonesArray* mPointsArray; ///< Array of MC hits
      TClonesArray* mDigitsArray; ///< Array of digits
      
      ClassDefOverride(DigitizerTask, 1)
	
    };    
  }
}

#endif
