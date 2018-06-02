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
/// \brief Task driving the convertion from Hit to Digit
/// \author bogdan.vulpescu@cern.ch
/// \date 03/05/2017

#include "MFTSimulation/DigitizerTask.h"
#include "ITSMFTSimulation/Hit.h"
#include "MFTBase/GeometryTGeo.h"
#include "MathUtils/Utils.h"

#include "FairLogger.h"      // for LOG
#include "FairRootManager.h" // for FairRootManager

ClassImp(o2::MFT::DigitizerTask);

using namespace o2::MFT;
using namespace o2::detectors;
using namespace o2::utils;

using o2::ITSMFT::DigiParams;

//_____________________________________________________________________________
DigitizerTask::DigitizerTask() : FairTask("MFTDigitizerTask"), mDigitizer() {}
//_____________________________________________________________________________
DigitizerTask::~DigitizerTask()
{
  mDigitsArray.clear();
  mMCTruthArray.clear();
}
/// \brief Init function
///
/// Inititializes the digitizer and connects input and output container
//_____________________________________________________________________________
InitStatus DigitizerTask::Init()
{
  FairRootManager* mgr = FairRootManager::Instance();
  if (!mgr) {
    LOG(ERROR) << "Could not instantiate FairRootManager. Exiting ..." << FairLogger::endl;
    return kERROR;
  }

  mHitsArray = mgr->InitObjectAs<const std::vector<o2::ITSMFT::Hit>*>("MFTHit");
  if (!mHitsArray) {
    LOG(ERROR) << "MFT hits not registered in the FairRootManager. Exiting ..." << FairLogger::endl;
    return kERROR;
  }

  // Register output container
  mgr->RegisterAny("MFTDigit", mDigitsArrayPtr, kTRUE);
  mgr->RegisterAny("MFTDigitMCTruth", mMCTruthArrayPtr, kTRUE);

  mDigitizer.setDigiParams(mParams);

  mDigitizer.setCoeffToNanoSecond(mFairTimeUnitInNS);

  GeometryTGeo* geom = GeometryTGeo::Instance();
  geom->fillMatrixCache(o2::utils::bit2Mask(o2::TransformType::L2G)); // make sure L2G matrices are loaded
  mDigitizer.setGeometry(geom);

  mDigitizer.setHits(mHitsArray);
  mDigitizer.setDigits(mDigitsArrayPtr);
  mDigitizer.setMCLabels(mMCTruthArrayPtr);

  mDigitizer.init();

  return kSUCCESS;
}

//_____________________________________________________________________________
void DigitizerTask::Exec(Option_t* option)
{
  FairRootManager* mgr = FairRootManager::Instance();

  mDigitsArray.clear();
  mMCTruthArray.clear();
  mDigitizer.setEventTime(mgr->GetEventTime());

  // the type of digitization is steered by the DigiParams object of the Digitizer
  LOG(DEBUG) << "Running digitization on new event " << mEventID << " from source " << mSourceID << FairLogger::endl;

  /// RS: ATTENTION: this is just a trick until we clarify how the hits from different source are
  /// provided and identified.
  mDigitizer.setCurrSrcID(mSourceID);
  mDigitizer.setCurrEvID(mEventID);

  mDigitizer.process();

  mEventID++;
}

//________________________________________________________
void DigitizerTask::FinishTask()
{
  // finalize digitization, if needed, flash remaining digits
  if (!mParams.isContinuous()) {
    return;
  }
  FairRootManager* mgr = FairRootManager::Instance();
  mgr->SetLastFill(kTRUE); /// necessary, otherwise the data is not written out
  mDigitsArray.clear();
  mMCTruthArray.clear();
  mDigitizer.fillOutputContainer();
}
