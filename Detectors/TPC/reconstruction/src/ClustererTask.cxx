// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file ClustererTask.cxx
/// \brief Implementation of the TPC Clusterer Task

#include "TPCReconstruction/ClustererTask.h"

#include "FairLogger.h"          // for LOG
#include "FairRootManager.h"     // for FairRootManager

#include "TPCBase/Digit.h"
#include "TPCReconstruction/HwClusterer.h"

ClassImp(o2::TPC::ClustererTask);

using namespace o2::TPC;

//_____________________________________________________________________
ClustererTask::ClustererTask(int sectorid)
  : FairTask("TPCClustererTask"),
    mIsContinuousReadout(true),
    mEventCount(0),
    mClusterSector(sectorid),
    mHwClusterer(nullptr),
    mDigitsArray(nullptr),
    mDigitMCTruthArray(nullptr),
    mDummy(nullptr),
    mHwClustersArray(nullptr),
    mHwClustersMCTruthArray(nullptr)
{
  // TODO: remove dummy
}

//_____________________________________________________________________
ClustererTask::~ClustererTask()
{
  LOG(DEBUG) << "Enter Destructor of ClustererTask" << FairLogger::endl;
  // TODO: remove dummy, then make destructor default
  if (mDummy)
    delete mDummy;
}
//_____________________________________________________________________
/// \brief Init function
/// Inititializes the clusterer and connects input and output container
InitStatus ClustererTask::Init()
{
  LOG(DEBUG) << "Enter Initializer of ClustererTask" << FairLogger::endl;

  FairRootManager *mgr = FairRootManager::Instance();
  if( !mgr ) {
    LOG(ERROR) << "Could not instantiate FairRootManager. Exiting ..." << FairLogger::endl;
    return kERROR;
  }

  if (mClusterSector < 0 || mClusterSector >= Sector::MAXSECTOR) {
    LOG(ERROR) << "Sector ID " << mClusterSector << " is not supported. Exiting ..." << FairLogger::endl;
    return kERROR;
  }

  // Register input container
  std::stringstream sectornamestr;
  sectornamestr << "TPCDigit" << mClusterSector;
  LOG(INFO) << "FETCHING DIGITS FOR SECTOR " << mClusterSector << "\n";
  mDigitsArray = std::unique_ptr<const std::vector<Digit>>(
    mgr->InitObjectAs<const std::vector<Digit>*>(sectornamestr.str().c_str()));
  if (!mDigitsArray) {
    LOG(ERROR) << "TPC points not registered in the FairRootManager. Exiting ..." << FairLogger::endl;
    return kERROR;
  }
  std::stringstream mcsectornamestr;
  mcsectornamestr << "TPCDigitMCTruth" << mClusterSector;
  mDigitMCTruthArray = std::unique_ptr<const MCLabelContainer>(
    mgr->InitObjectAs<const MCLabelContainer*>(mcsectornamestr.str().c_str()));
  if (!mDigitMCTruthArray) {
    LOG(ERROR) << "TPC MC Truth not registered in the FairRootManager. Exiting ..." << FairLogger::endl;
    return kERROR;
  }

  // Register output container
  // TODO: remove dummy
  // first register a dummy plain pointer that the dictionary of the vector is found
  mDummy = new std::vector<ClusterHardwareContainer8kb>();
  mgr->RegisterAny("dummy", mDummy, kTRUE);
  mHwClustersArray = std::make_shared<std::vector<ClusterHardwareContainer8kb>>();
  // then using the trick to register the shared pointer with FairRootManager
  static auto clusterArrayTmpPtr = mHwClustersArray.get();
  mgr->RegisterAny(Form("TPCClusterHW%i", mClusterSector), clusterArrayTmpPtr, kTRUE);

  // Register MC Truth output container
  mHwClustersMCTruthArray = std::make_shared<MCLabelContainer>();
  // a trick to register the shared pointer with FairRootManager
  static auto clusterMcTruthTmpPtr = mHwClustersMCTruthArray.get();
  mgr->RegisterAny(Form("TPCClusterHWMCTruth%i", mClusterSector), clusterMcTruthTmpPtr, kTRUE);

  // create clusterer and pass output pointer
  mHwClusterer = std::make_unique<HwClusterer>(mHwClustersArray, mHwClustersMCTruthArray, mClusterSector);
  mHwClusterer->setContinuousReadout(mIsContinuousReadout);

  // TODO: implement noise/pedestal objects
  //    mHwClusterer->setNoiseObject(...);
  //    mHwClusterer->setPedestalObject(...);

  return kSUCCESS;
}

//_____________________________________________________________________
void ClustererTask::Exec(Option_t *option)
{
  LOG(DEBUG) << "Running clusterization on event " << mEventCount << " with " << mDigitsArray->size() << " digits." << FairLogger::endl;

  if (mHwClustersArray)
    mHwClustersArray->clear();
  if (mHwClustersMCTruthArray)
    mHwClustersMCTruthArray->clear();

  mHwClusterer->Process(*mDigitsArray.get(), *mDigitMCTruthArray.get(), mEventCount);
  LOG(DEBUG) << "Hw clusterer delivered " << mHwClustersArray->size() << " cluster container" << FairLogger::endl
             << FairLogger::endl;

  ++mEventCount;
}

//_____________________________________________________________________
void ClustererTask::FinishTask()
{
  LOG(DEBUG) << "Finish clusterization" << FairLogger::endl;

  if (mHwClustersArray)
    mHwClustersArray->clear();
  if (mHwClustersMCTruthArray)
    mHwClustersMCTruthArray->clear();

  mHwClusterer->FinishProcess(*mDigitsArray.get(), *mDigitMCTruthArray.get(), mEventCount);
  LOG(DEBUG) << "Hw clusterer delivered " << mHwClustersArray->size() << " cluster container" << FairLogger::endl
             << FairLogger::endl;

  ++mEventCount;
}
