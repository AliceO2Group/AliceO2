// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file DigitizerTask.cxx
/// \brief Implementation of the ITS digitizer task

//
//  Created by Markus Fasel on 16.07.15.
//
//

#include "ITSSimulation/DigitizerTask.h"

#include "FairLogger.h"      // for LOG
#include "FairRootManager.h" // for FairRootManager
#include "TClonesArray.h"    // for TClonesArray
#include "TObject.h"         // for TObject


ClassImp(o2::ITS::DigitizerTask)

using namespace o2::ITS;
using namespace o2::ITSMFT;

DigitizerTask::DigitizerTask(Bool_t useAlpide)
  : FairTask("ITSDigitizerTask"), mUseAlpideSim(useAlpide), mDigitizer()
{
}

DigitizerTask::~DigitizerTask()
{
  if (mDigitsArray) {
    mDigitsArray->Delete();
    delete mDigitsArray;
  }
}

/// \brief Init function
///
/// Inititializes the digitizer and connects input and output container
InitStatus DigitizerTask::Init()
{
  FairRootManager* mgr = FairRootManager::Instance();
  if (!mgr) {
    LOG(ERROR) << "Could not instantiate FairRootManager. Exiting ..." << FairLogger::endl;
    return kERROR;
  }

  mPointsArray = dynamic_cast<TClonesArray*>(mgr->GetObject("ITSPoint"));
  if (!mPointsArray) {
    LOG(ERROR) << "ITS points not registered in the FairRootManager. Exiting ..." << FairLogger::endl;
    return kERROR;
  }

  // Register output container
  mDigitsArray = new TClonesArray("o2::ITSMFT::Digit");
  mgr->Register("ITSDigit", "ITS", mDigitsArray, kTRUE);

  DigiParams param; // RS: TODO: Eventually load this from the CCDB

  param.setContinuous(mContinuous);
  param.setPointDigitsMethod(mUseAlpideSim ? DigiParams::p2dCShape : DigiParams::p2dSimple);
  mDigitizer.setDigiParams(param);

  mDigitizer.setCoeffToNanoSecond(mFairTimeUnitInNS);
  
  mDigitizer.init(kTRUE);

  return kSUCCESS;
}

//________________________________________________________
void DigitizerTask::Exec(Option_t* option)
{
  FairRootManager* mgr = FairRootManager::Instance();

  mDigitsArray->Clear();
  mDigitizer.setEventTime(mgr->GetEventTime());

  // the type of digitization is steered by the DigiParams object of the Digitizer
  LOG(DEBUG) << "Running digitization on new event" << FairLogger::endl;
  mDigitizer.process(mPointsArray,mDigitsArray);
}

//________________________________________________________
void DigitizerTask::FinishTask()
{
  // finalize digitization, if needed, flash remaining digits
  if(!mContinuous) return;
  FairRootManager *mgr = FairRootManager::Instance();
  mgr->SetLastFill(kTRUE); /// necessary, otherwise the data is not written out
  mDigitsArray->Clear();
  mDigitizer.fillOutputContainer(mDigitsArray);
}
