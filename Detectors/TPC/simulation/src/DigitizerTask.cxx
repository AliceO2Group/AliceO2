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
#include "TH3.h"
#include "TRandom.h"
#include "TTree.h"

#include "TPCBase/Digit.h"
#include "TPCBase/Sector.h"
#include "TPCSimulation/DigitContainer.h"
#include "TPCSimulation/DigitMCMetaData.h"
#include "TPCSimulation/Digitizer.h"
#include "TPCSimulation/DigitizerTask.h"
#include "TPCSimulation/Point.h"
#include "TPCSimulation/SAMPAProcessing.h"
#include "TPCBase/CDBInterface.h"

#include "FairLogger.h"
#include "FairRootManager.h"

#include <sstream>

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
    mProcessTimeChunks(false),
    mIsContinuousReadout(true),
    mDigitDebugOutput(false),
    mHitSector(sectorid),
    mStartTime(-1),
    mEndTime(-1)
{
  mDigitizer = new Digitizer;
}

DigitizerTask::~DigitizerTask()
{
  delete mDigitizer;
  // We need to clarify the ownsership of these potentially external containers
  // and reenable the cleanup
  // delete mDigitsArray;
  // delete mDigitsDebugArray;
}

InitStatus DigitizerTask::Init()
{
  /// Initialize the task and the input and output containers
  FairRootManager* mgr = FairRootManager::Instance();
  if (!mgr) {
    LOG(ERROR) << "Could not instantiate FairRootManager. Exiting ..." << FairLogger::endl;
    return kERROR;
  }

  /// For the time being use the defaults for the CDB
  auto& cdb = CDBInterface::instance();
  cdb.setUseDefaults();

  /// Fetch the hits for the sector which is to be processed
  LOG(DEBUG) << "Processing sector " << mHitSector << "  - loading HitSector "
             << int(Sector::getLeft(Sector(mHitSector))) << " and " << mHitSector << "\n";
  std::stringstream sectornamestrleft, sectornamestrright;
  sectornamestrleft << "TPCHitsShiftedSector" << int(Sector::getLeft(Sector(mHitSector)));
  sectornamestrright << "TPCHitsShiftedSector" << mHitSector;
  mSectorHitsArrayLeft = mgr->InitObjectAs<const std::vector<HitGroup>*>(sectornamestrleft.str().c_str());
  mSectorHitsArrayRight = mgr->InitObjectAs<const std::vector<HitGroup>*>(sectornamestrright.str().c_str());

  // Register output container
  mDigitsArray = new std::vector<Digit>;
  mgr->RegisterAny(Form("TPCDigit%i", mHitSector), mDigitsArray, kTRUE);

  // Register MC Truth container
  mMCTruthArray = new typename std::remove_pointer<decltype(mMCTruthArray)>::type;
  mgr->RegisterAny(Form("TPCDigitMCTruth%i", mHitSector), mMCTruthArray, kTRUE);

  // Register additional (optional) debug output
  if (mDigitDebugOutput) {
    mDigitsDebugArray = new std::vector<DigitMCMetaData>;
    mgr->RegisterAny(Form("TPCDigitMCMetaData%i", mHitSector), mDigitsDebugArray, kTRUE);
  }

  mDigitizer->init();
  mDigitContainer = mDigitizer->getDigitContainer();
  mDigitContainer->setup(mHitSector);
  return kSUCCESS;
}

InitStatus DigitizerTask::Init2()
{
  /// For the time being use the defaults for the CDB
  auto& cdb = CDBInterface::instance();
  cdb.setUseDefaults();

  mDigitizer->init();
  mDigitContainer = mDigitizer->getDigitContainer();
  return kSUCCESS;
}

void DigitizerTask::Exec(Option_t* option)
{
  FairRootManager* mgr = FairRootManager::Instance();

  // time should be given in us
  float eventTime = static_cast<float>(mgr->GetEventTime() * 0.001);
  if (mEventTimes.size()) {
    eventTime = mEventTimes[mCurrentEvent++];
    LOG(DEBUG) << "Event time taken from bunch simulation";
  }

  static SAMPAProcessing& sampaProcessing = SAMPAProcessing::instance();
  const int eventTimeBin = sampaProcessing.getTimeBinFromTime(eventTime);

  LOG(DEBUG) << "Running digitization for sector " << mHitSector << " on new event at time " << eventTime
             << " us in time bin " << eventTimeBin << FairLogger::endl;
  mDigitsArray->clear();
  mMCTruthArray->clear();
  if (mDigitDebugOutput) {
    mDigitsDebugArray->clear();
  }

  // Treat the chosen sector
  mDigitContainer = mDigitizer->Process(Sector(mHitSector), *mSectorHitsArrayLeft, mgr->GetEntryNr(), eventTime);
  mDigitContainer = mDigitizer->Process(Sector(mHitSector), *mSectorHitsArrayRight, mgr->GetEntryNr(), eventTime);
  mDigitContainer->fillOutputContainer(mDigitsArray, *mMCTruthArray, mDigitsDebugArray, eventTimeBin,
                                       mIsContinuousReadout);
}

void DigitizerTask::Exec2(Option_t* option)
{
  mDigitsArray->clear();
  mMCTruthArray->clear();
  if (mDigitDebugOutput) {
    mDigitsDebugArray->clear();
  }

  const auto sec = Sector(mHitSector);
  static SAMPAProcessing& sampaProcessing = SAMPAProcessing::instance();
  const int endTimeBin = sampaProcessing.getTimeBinFromTime(mEndTime * 0.001f);
  if (mProcessTimeChunks) {
    mDigitContainer->setFirstTimeBin(sampaProcessing.getTimeBinFromTime(mStartTime * 0.001f));
  }
  mDigitContainer = mDigitizer->Process2(sec, *mAllSectorHitsLeft, *mHitIdsLeft, *mRunContext);
  mDigitContainer = mDigitizer->Process2(sec, *mAllSectorHitsRight, *mHitIdsRight, *mRunContext);
  mDigitContainer->fillOutputContainer(mDigitsArray, *mMCTruthArray, mDigitsDebugArray, endTimeBin,
                                       mIsContinuousReadout);
}

void DigitizerTask::FinishTask()
{
  if (!mIsContinuousReadout)
    return;
  FairRootManager* mgr = FairRootManager::Instance();
  mgr->SetLastFill(kTRUE); /// necessary, otherwise the data is not written out
  mDigitsArray->clear();
  mMCTruthArray->clear();
  if (mDigitDebugOutput) {
    mDigitsDebugArray->clear();
  }
  mDigitContainer->fillOutputContainer(mDigitsArray, *mMCTruthArray, mDigitsDebugArray, mTimeBinMax,
                                       mIsContinuousReadout);
}

void DigitizerTask::FinishTask2()
{
  if (!mIsContinuousReadout)
    return;
  mDigitContainer->fillOutputContainer(mDigitsArray, *mMCTruthArray, mDigitsDebugArray, mTimeBinMax,
                                       mIsContinuousReadout);
}

void DigitizerTask::initBunchTrainStructure(const size_t numberOfEvents)
{
  LOG(DEBUG) << "Initialising bunch train structure for " << numberOfEvents << " evnets" << FairLogger::endl;
  // Parameters for bunches
  const double abortGap = 3e-6; //
  const double collFreq = 50e3;
  const double bSpacing = 50e-9; // bunch spacing
  const int nTrainBunches = 48;
  const int nTrains = 12;
  const double revFreq = 1.11e4; // revolution frequency
  const double collProb = collFreq / (nTrainBunches * nTrains * revFreq);
  const double trainLength = bSpacing * (nTrainBunches - 1);
  const double totTrainLength = nTrains * trainLength;
  const double trainSpacing = (1. / revFreq - abortGap - totTrainLength) / (nTrains - 1);

  // counters
  double eventTime = 0.; // all in seconds
  size_t nGeneratedEvents = 0;
  size_t bunchCounter = 0;
  size_t trainCounter = 0;

  // reset vector
  mEventTimes.clear();

  while (nGeneratedEvents < numberOfEvents + 2) {
    //  std::cout <<trainCounter << " " << bunchCounter << " "<< "eventTime " << eventTime << std::endl;

    int nCollsInCrossing = gRandom->Poisson(collProb);
    for (int iColl = 0; iColl < nCollsInCrossing; iColl++) {
      // printf("Generating event %3d (%.3g)\n",nGeneratedEvents,eventTime);
      mEventTimes.emplace_back(static_cast<float>(eventTime * 1e6)); // convert to us
      nGeneratedEvents++;
    }
    bunchCounter++;

    if (bunchCounter >= nTrainBunches) {

      trainCounter++;
      if (trainCounter >= nTrains) {
        eventTime += abortGap;
        trainCounter = 0;
      } else
        eventTime += trainSpacing;

      bunchCounter = 0;
    } else
      eventTime += bSpacing;
  }
}

void DigitizerTask::enableSCDistortions(SpaceCharge::SCDistortionType distortionType, TH3 *hisInitialSCDensity, int nZSlices, int nPhiBins, int nRBins)
{
  if (distortionType==SpaceCharge::SCDistortionType::SCDistortionsConstant){
    LOG(INFO) << "Using constant space-charge distortions." << FairLogger::endl;
    if (hisInitialSCDensity==nullptr) LOG(FATAL) << "Constant space-charge distortions require an initial space-charge density histogram. Please provide the path to the root file (O2TPCSCDensityHisFilePath) and the histogram name (O2TPCSCDensityHisName) in your environment variables." << FairLogger::endl;
  }
  if (distortionType==SpaceCharge::SCDistortionType::SCDistortionsRealistic) LOG(INFO) << "Using realistic space-charge distortions." << FairLogger::endl;
  if (hisInitialSCDensity) LOG(INFO) << "Providing initial space-charge density histogram: " << hisInitialSCDensity->GetName() << FairLogger::endl;
  mDigitizer->enableSCDistortions(distortionType, hisInitialSCDensity, nZSlices, nPhiBins, nRBins);
}
