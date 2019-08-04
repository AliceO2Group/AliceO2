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

#include "FairLogger.h"      // for LOG
#include "FairRootManager.h" // for FairRootManager

ClassImp(o2::tpc::ClustererTask);

using namespace o2::tpc;

//_____________________________________________________________________
ClustererTask::ClustererTask(int sectorid)
  : FairTask("TPCClustererTask"),
    mClusterSector(sectorid),
    mHwClusterer(),
    mDigitsArray(),
    mDigitMCTruthArray(),
    mHwClustersArray(),
    mHwClustersMCTruthArray()
{
}

//_____________________________________________________________________
/// \brief Init function
/// Inititializes the clusterer and connects input and output container
InitStatus ClustererTask::Init()
{
  LOG(DEBUG) << "Enter Initializer of ClustererTask" << FairLogger::endl;

  FairRootManager* mgr = FairRootManager::Instance();
  if (!mgr) {
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
  mHwClustersArray = std::make_unique<std::vector<OutputType>>();
  // a trick to register the unique pointer with FairRootManager
  static auto clusterArrayTmpPtr = mHwClustersArray.get();
  mgr->RegisterAny(Form("TPCClusterHW%i", mClusterSector), clusterArrayTmpPtr, kTRUE);

  // Register MC Truth output container
  mHwClustersMCTruthArray = std::make_unique<MCLabelContainer>();
  // a trick to register the unique pointer with FairRootManager
  static auto clusterMcTruthTmpPtr = mHwClustersMCTruthArray.get();
  mgr->RegisterAny(Form("TPCClusterHWMCTruth%i", mClusterSector), clusterMcTruthTmpPtr, kTRUE);

  // create clusterer and pass output pointer
  mHwClusterer = std::make_unique<HwClusterer>(mHwClustersArray.get(), mClusterSector, mHwClustersMCTruthArray.get());
  mHwClusterer->setContinuousReadout(mIsContinuousReadout);

  // TODO: implement noise/pedestal objects
  //    mHwClusterer->setNoiseObject(...);
  //    mHwClusterer->setPedestalObject(...);

  return kSUCCESS;
}

//_____________________________________________________________________
void ClustererTask::Exec(Option_t* option)
{
  LOG(DEBUG) << "Running clusterization on event " << mEventCount << " with " << mDigitsArray->size() << " digits." << FairLogger::endl;

  if (mHwClustersArray)
    mHwClustersArray->clear();
  if (mHwClustersMCTruthArray)
    mHwClustersMCTruthArray->clear();

  mHwClusterer->process(*mDigitsArray.get(), mDigitMCTruthArray.get());
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

  mHwClusterer->finishProcess(*mDigitsArray.get(), mDigitMCTruthArray.get());
  LOG(DEBUG) << "Hw clusterer delivered " << mHwClustersArray->size() << " cluster container" << FairLogger::endl
             << FairLogger::endl;

  ++mEventCount;
}
