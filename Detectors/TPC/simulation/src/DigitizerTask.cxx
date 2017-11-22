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
#include "TRandom.h"

#include "TPCSimulation/DigitizerTask.h"
#include "TPCSimulation/DigitContainer.h"
#include "TPCSimulation/Digitizer.h"
#include "TPCSimulation/Point.h"
#include "TPCBase/Sector.h"
#include "TPCBase/Digit.h"
#include "TPCSimulation/DigitMCMetaData.h"

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
    mMCTruthArray(nullptr),
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
    mSectorHitsArray[mHitSector] = mgr->InitObjectAs<const std::vector<HitGroup>*>(sectornamestr.str().c_str());
  }
  else {
    // in case we are treating all sectors
    for (int s=0;s<Sector::MAXSECTOR;++s){
      std::stringstream sectornamestr;
      sectornamestr << "TPCHitsSector" << s;
      LOG(INFO) << "FETCHING HITS FOR SECTOR " << s << "\n";
      mSectorHitsArray[s] = mgr->InitObjectAs<const std::vector<HitGroup>*>(sectornamestr.str().c_str());
    }
  }
  
  // Register output container
  mDigitsArray = new std::vector<o2::TPC::Digit>;
  mgr->RegisterAny("TPCDigit", mDigitsArray, kTRUE);

  // Register MC Truth container
  mMCTruthArray = new typename std::remove_pointer<decltype(mMCTruthArray)>::type;
  mgr->RegisterAny("TPCDigitMCTruth", mMCTruthArray, kTRUE);

  // Register additional (optional) debug output
  if(mDigitDebugOutput) {
    mDigitsDebugArray = new std::vector<o2::TPC::DigitMCMetaData>;
    mgr->RegisterAny("TPCDigitMCMetaData", mDigitsDebugArray, kTRUE);
  }
  
  mDigitizer->init();
  mDigitContainer = mDigitizer->getDigitContainer();
  return kSUCCESS;
}

void DigitizerTask::Exec(Option_t *option)
{
  FairRootManager *mgr = FairRootManager::Instance();

  // time should be given in us
  float eventTime = static_cast<float>(mgr->GetEventTime() * 0.001);
  if (mEventTimes.size()) {
    eventTime = mEventTimes[mCurrentEvent++];
    LOG(DEBUG) << "Event time taken from bunch simulation";
  }
  const int eventTimeBin = Digitizer::getTimeBinFromTime(eventTime);

  LOG(DEBUG) << "Running digitization on new event at time " << eventTime << " us in time bin " << eventTimeBin << FairLogger::endl;
  mDigitsArray->clear();
  mMCTruthArray->clear();
  if(mDigitDebugOutput) {
    mDigitsDebugArray->clear();
  }

  if (mHitSector == -1){
    // treat all sectors
    for (int s=0; s<Sector::MAXSECTOR; ++s){
      LOG(DEBUG) << "Processing sector " << s << "\n";
      mDigitContainer = mDigitizer->Process(*mSectorHitsArray[s], eventTime);
    }
  }
  else {
    // treat only chosen sector
    mDigitContainer = mDigitizer->Process(*mSectorHitsArray[mHitSector], eventTime);
  }
  mDigitContainer->fillOutputContainer(mDigitsArray, *mMCTruthArray, mDigitsDebugArray, eventTimeBin, mIsContinuousReadout);
}

void DigitizerTask::FinishTask()
{
  if(!mIsContinuousReadout) return;
  FairRootManager *mgr = FairRootManager::Instance();
  mgr->SetLastFill(kTRUE); /// necessary, otherwise the data is not written out
  mDigitsArray->clear();
  mMCTruthArray->clear();
  if(mDigitDebugOutput) {
    mDigitsDebugArray->clear();
  }
  mDigitContainer->fillOutputContainer(mDigitsArray, *mMCTruthArray, mDigitsDebugArray, mTimeBinMax, mIsContinuousReadout);
}

void DigitizerTask::initBunchTrainStructure(const size_t numberOfEvents)
{
  LOG(DEBUG) << "Initialising bunch train structure for " << numberOfEvents << " evnets" << FairLogger::endl;
  // Parameters for bunches
  const double abortGap = 3e-6; //
  const double collFreq = 50e3;
  const double bSpacing = 50e-9; //bunch spacing
  const int nTrainBunches = 48;
  const int nTrains = 12;
  const double revFreq = 1.11e4; //revolution frequency
  const double collProb = collFreq/(nTrainBunches*nTrains*revFreq);
  const double trainLength = bSpacing*(nTrainBunches-1);
  const double totTrainLength = nTrains*trainLength;
  const double trainSpacing = (1./revFreq - abortGap - totTrainLength)/(nTrains-1); 

  // counters
  double eventTime=0.; // all in seconds
  size_t nGeneratedEvents = 0;
  size_t bunchCounter = 0;
  size_t trainCounter = 0;

  // reset vector
  mEventTimes.clear();

  while (nGeneratedEvents < numberOfEvents+2){
    //  std::cout <<trainCounter << " " << bunchCounter << " "<< "eventTime " << eventTime << std::endl;

    int nCollsInCrossing = gRandom -> Poisson(collProb);
    for(int iColl = 0; iColl<nCollsInCrossing; iColl++){
      //printf("Generating event %3d (%.3g)\n",nGeneratedEvents,eventTime);
      mEventTimes.emplace_back(static_cast<float>(eventTime*1e6)); // convert to us
      nGeneratedEvents++;
    }
    bunchCounter++;

    if(bunchCounter>=nTrainBunches){

      trainCounter++;
      if(trainCounter>=nTrains){
        eventTime+=abortGap;
        trainCounter=0;
      }
      else eventTime+=trainSpacing;

      bunchCounter=0;
    }
    else eventTime+= bSpacing;

  }
}
