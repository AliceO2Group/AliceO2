// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "EMCALSimulation/DigitizerTask.h"
#include "EMCALBase/Geometry.h"
#include "EMCALBase/GeometryBase.h"
#include "EMCALBase/Hit.h"
#include "EMCALSimulation/Digitizer.h"

#include "FairLogger.h"      // for LOG
#include "FairRootManager.h" // for FairRootManager
#include "FairTask.h"        // for FairTask, InitStatus
#include "Rtypes.h"          // for DigitizerTask::Class, ClassDef, etc

ClassImp(o2::emcal::DigitizerTask);

using o2::emcal::Digit;
using o2::emcal::Digitizer;
using o2::emcal::Hit;

using namespace o2::emcal;

DigitizerTask::DigitizerTask() : FairTask("EMCALDigitizerTask"), mDigitizer() {}

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
    LOG(ERROR) << "Could not instantiate FairRootManager. Exiting ...";
    return kERROR;
  }

  mHitsArray = mgr->InitObjectAs<const std::vector<o2::emcal::Hit>*>("EMCALHit");
  if (!mHitsArray) {
    LOG(ERROR) << "EMCAL hits not registered in the FairRootManager. Exiting ...";
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

  if (mDigitsArray)
    mDigitsArray->clear();
  mDigitizer.setEventTime(mgr->GetEventTime());

  LOG(DEBUG) << "Running digitization on new event " << mEventID << " from source " << mSourceID;

  mDigitizer.setCurrSrcID(mSourceID);
  mDigitizer.setCurrEvID(mEventID);

  mDigitizer.process(*mHitsArray, *mDigitsArray);

  mEventID++;
}

//________________________________________________________
void DigitizerTask::FinishTask()
{
  // finalize digitization, if needed, flash remaining digits
  FairRootManager* mgr = FairRootManager::Instance();
  mgr->SetLastFill(kTRUE); /// necessary, otherwise the data is not written out
  if (mDigitsArray)
    mDigitsArray->clear();
  // mDigitizer.fillOutputContainer(mDigitsArray);
  mDigitizer.finish();
}
