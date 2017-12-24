// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "EMCALBase/Hit.h"
#include "EMCALBase/Geometry.h"
#include "EMCALBase/GeometryBase.h"
#include "EMCALSimulation/Digitizer.h"
#include "EMCALSimulation/DigitizerTask.h"

#include "FairTask.h" // for FairTask, InitStatus
#include "FairLogger.h"      // for LOG
#include "FairRootManager.h" // for FairRootManager
#include "Rtypes.h"   // for DigitizerTask::Class, ClassDef, etc

ClassImp(o2::EMCAL::DigitizerTask);

using o2::EMCAL::Hit;
using o2::EMCAL::Digit;
using o2::EMCAL::Digitizer;

using namespace o2::EMCAL;

DigitizerTask::DigitizerTask()
  : FairTask("EMCALDigitizerTask"), mDigitizer()
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

  mHitsArray = mgr->InitObjectAs<const std::vector<o2::EMCAL::Hit>*>("EMCALHit");
  if (!mHitsArray) {
    LOG(ERROR) << "EMCAL hits not registered in the FairRootManager. Exiting ..." << FairLogger::endl;
    return kERROR;
  }

  // Register output container
  mgr->RegisterAny("EMCALDigit", mDigitsArray, kTRUE);

  mDigitizer.setCoeffToNanoSecond(mFairTimeUnitInNS);
  
  Geometry* geom = Geometry::GetInstance();
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

  LOG(DEBUG) << "Running digitization on new event " << mEventID
	     << " from source " << mSourceID << FairLogger::endl;

  mDigitizer.setCurrSrcID( mSourceID );
  mDigitizer.setCurrEvID( mEventID );
  
  mDigitizer.process(mHitsArray,mDigitsArray);

  mEventID++;
}

//________________________________________________________
void DigitizerTask::FinishTask()
{
  // finalize digitization, if needed, flash remaining digits
  FairRootManager *mgr = FairRootManager::Instance();
  mgr->SetLastFill(kTRUE); /// necessary, otherwise the data is not written out
  if (mDigitsArray) mDigitsArray->clear();
  //mDigitizer.fillOutputContainer(mDigitsArray);
  mDigitizer.finish();
}
