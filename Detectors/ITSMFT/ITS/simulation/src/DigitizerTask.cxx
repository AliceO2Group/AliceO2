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
#include "ITSMFTSimulation/Hit.h"
#include "DetectorsBase/Utils.h"
#include "ITSBase/GeometryTGeo.h"

#include "FairLogger.h"      // for LOG
#include "FairRootManager.h" // for FairRootManager


ClassImp(o2::ITS::DigitizerTask)

using namespace o2::ITS;
using namespace o2::Base;
using namespace o2::Base::Utils;

using o2::ITSMFT::DigiParams;

DigitizerTask::DigitizerTask(Bool_t useAlpide)
  : FairTask("ITSDigitizerTask"), mUseAlpideSim(useAlpide), mDigitizer()
{
}

DigitizerTask::~DigitizerTask()
{
  if (mDigitsArray) {
    mDigitsArray->clear();
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

  mHitsArray = mgr->InitObjectAs<const std::vector<o2::ITSMFT::Hit>*>("ITSHit");
  if (!mHitsArray) {
    LOG(ERROR) << "ITS hits not registered in the FairRootManager. Exiting ..." << FairLogger::endl;
    return kERROR;
  }

  // Register output container
  mgr->RegisterAny("ITSDigit", mDigitsArray, kTRUE);

  DigiParams param; // RS: TODO: Eventually load this from the CCDB

  param.setContinuous(mContinuous);
  param.setHitDigitsMethod(mUseAlpideSim ? DigiParams::p2dCShape : DigiParams::p2dSimple);
  mDigitizer.setDigiParams(param);

  mDigitizer.setCoeffToNanoSecond(mFairTimeUnitInNS);
  
  GeometryTGeo* geom = GeometryTGeo::Instance();
  geom->fillMatrixCache( bit2Mask(TransformType::L2G) ); // make sure L2G matrices are loaded
  mDigitizer.setGeometry(geom);
  
  mDigitizer.init();

  return kSUCCESS;
}

//________________________________________________________
void DigitizerTask::Exec(Option_t* option)
{
  FairRootManager* mgr = FairRootManager::Instance();

  if (mDigitsArray) mDigitsArray->clear();
  mDigitizer.setEventTime(mgr->GetEventTime());

  // the type of digitization is steered by the DigiParams object of the Digitizer
  LOG(DEBUG) << "Running digitization on new event " << mEventID
             << " from source " << mSourceID << FairLogger::endl;

  /// RS: ATTENTION: this is just a trick until we clarify how the hits from different source are
  /// provided and identified.
  mDigitizer.setCurrSrcID( mSourceID );
  mDigitizer.setCurrEvID( mEventID );
  
  mDigitizer.process(mHitsArray,mDigitsArray);

  mEventID++;
}

//________________________________________________________
void DigitizerTask::FinishTask()
{
  // finalize digitization, if needed, flash remaining digits
  if(!mContinuous) return;
  FairRootManager *mgr = FairRootManager::Instance();
  mgr->SetLastFill(kTRUE); /// necessary, otherwise the data is not written out
  if (mDigitsArray) mDigitsArray->clear();
  mDigitizer.fillOutputContainer(mDigitsArray);
}
