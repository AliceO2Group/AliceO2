// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file Digitizer.cxx
/// \brief Implementation of the ALICE TPC digitizer task
/// \author Andi Mathis, TU München, andreas.mathis@ph.tum.de

#include "TFile.h"
#include "TTree.h"

#include "TPCSimulation/DigitizerTask.h"
#include "TPCSimulation/DigitContainer.h"
#include "TPCSimulation/Digitizer.h"
#include "TPCSimulation/Point.h"
#include "TPCBase/Sector.h"

#include "FairLogger.h"
#include "FairRootManager.h"

#include <sstream>
//#include "valgrind/callgrind.h"

ClassImp(o2::TPC::DigitizerTask)

using namespace o2::TPC;


DigitizerTask::DigitizerTask(int sectorid)
  : FairTask("TPCDigitizerTask"),
    mDigitizer(nullptr),
    mDigitContainer(nullptr),
    mDigitsArray(nullptr),
    mMCTruthArray(),
    mDigitsDebugArray(nullptr),
    mTimeBinMax(1000000),
    mIsContinuousReadout(true),
    mDigitDebugOutput(false),
    mHitSector(sectorid)
{
  /// \todo get rid of new
  mDigitizer = new Digitizer;
  //CALLGRIND_START_INSTRUMENTATION;
}

DigitizerTask::~DigitizerTask()
{
  delete mDigitizer;
  delete mDigitsArray;
  delete mDigitsDebugArray;

  //CALLGRIND_STOP_INSTRUMENTATION;
  //CALLGRIND_DUMP_STATS;
}


InitStatus DigitizerTask::Init()
{
  /// Initialize the task and the input and output containers
  FairRootManager *mgr = FairRootManager::Instance();
  if(!mgr){
    LOG(ERROR) << "Could not instantiate FairRootManager. Exiting ..." << FairLogger::endl;
    return kERROR;
  }

  // in case we are treating a specific sector
  if (mHitSector != -1){
    std::stringstream sectornamestr;
    sectornamestr << "TPCHitsSector" << mHitSector;
    LOG(INFO) << "FETCHING HITS FOR SECTOR " << mHitSector << "\n";
    mSectorHitsArray[mHitSector] = dynamic_cast<TClonesArray *>(mgr->GetObject(sectornamestr.str().c_str()));
  }
  else {
    // in case we are treating all sectors
    for (int s=0;s<Sector::MAXSECTOR;++s){
      std::stringstream sectornamestr;
      sectornamestr << "TPCHitsSector" << s;
      LOG(INFO) << "FETCHING HITS FOR SECTOR " << s << "\n";
      mSectorHitsArray[s] = dynamic_cast<TClonesArray *>(mgr->GetObject(sectornamestr.str().c_str()));
    }
  }
  
  // Register output container
  mDigitsArray = new TClonesArray("o2::TPC::Digit");
  mDigitsArray->BypassStreamer(true);
  mgr->Register("TPCDigit", "TPC", mDigitsArray, kTRUE);

  // Register MC Truth container
  mgr->Register("TPCDigitMCTruth", "TPC", &mMCTruthArray, kTRUE);

  // Register additional (optional) debug output
  if(mDigitDebugOutput) {
    mDigitsDebugArray = new TClonesArray("o2::TPC::DigitMCMetaData");
    mDigitsDebugArray->BypassStreamer(true);
    mgr->Register("TPCDigitMCMetaData", "TPC", mDigitsDebugArray, kTRUE);
  }
  
  mDigitizer->init();
  mDigitContainer = mDigitizer->getDigitContainer();
  return kSUCCESS;
}

void DigitizerTask::Exec(Option_t *option)
{
  FairRootManager *mgr = FairRootManager::Instance();

  const int eventTime = Digitizer::getTimeBinFromTime(mgr->GetEventTime() * 0.001);

  LOG(DEBUG) << "Running digitization on new event at time bin " << eventTime << FairLogger::endl;
  mDigitsArray->Delete();
  mMCTruthArray.clear();
  if(mDigitDebugOutput) {
    mDigitsDebugArray->Delete();
  }

  if (mHitSector == -1){
    // treat all sectors
    for (int s=0; s<Sector::MAXSECTOR; ++s){
      LOG(DEBUG) << "Processing sector " << s << "\n";
      mDigitContainer = mDigitizer->Process(mSectorHitsArray[s]);
    }
  }
  else {
    // treat only chosen sector
    mDigitContainer = mDigitizer->Process(mSectorHitsArray[mHitSector]);
  }
  mDigitContainer->fillOutputContainer(mDigitsArray, mMCTruthArray, mDigitsDebugArray, eventTime, mIsContinuousReadout);
}

void DigitizerTask::FinishTask()
{
  if(!mIsContinuousReadout) return;
  FairRootManager *mgr = FairRootManager::Instance();
  mgr->SetLastFill(kTRUE); /// necessary, otherwise the data is not written out
  mDigitsArray->Delete();
  mMCTruthArray.clear();
  if(mDigitDebugOutput) {
    mDigitsDebugArray->Delete();
  }
  mDigitContainer->fillOutputContainer(mDigitsArray, mMCTruthArray, mDigitsDebugArray, mTimeBinMax, mIsContinuousReadout);
}
