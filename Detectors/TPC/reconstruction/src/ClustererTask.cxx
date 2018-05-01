// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

//
//  ClustererTask.cxx
//  ALICEO2
//
//  Based on DigitizerTask
//
//

#include "TPCReconstruction/ClustererTask.h"
#include "DataFormatsTPC/Cluster.h"
#include "TPCBase/Digit.h"
#include "FairLogger.h"          // for LOG
#include "FairRootManager.h"     // for FairRootManager

ClassImp(o2::TPC::ClustererTask);

using namespace o2::TPC;

//_____________________________________________________________________
ClustererTask::ClustererTask(int sectorid)
  : FairTask("TPCClustererTask"),
    mIsContinuousReadout(true),
    mEventCount(0),
    mClusterSector(sectorid),
    mHwClusterer(nullptr),
    mDigitsArray(),
    mDigitMCTruthArray(),
    mHwClustersArray(nullptr),
    mHwClustersMCTruthArray(nullptr)
{
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
    LOG(ERROR) << "Sector ID " << mClusterSector << " is not supported, Exiting ..." << FairLogger::endl;
    return kERROR;
  }

  if (mClusterSector != -1) {
    std::stringstream sectornamestr;
    std::stringstream mcsectornamestr;
    sectornamestr << "TPCDigit" << mClusterSector;
    LOG(INFO) << "FETCHING DIGITS FOR SECTOR " << mClusterSector << "\n";
    mDigitsArray[mClusterSector] = mgr->InitObjectAs<const std::vector<Digit>*>(sectornamestr.str().c_str());
    if (!mDigitsArray[mClusterSector]) {
      LOG(ERROR) << "TPC points not registered in the FairRootManager. Exiting ..." << FairLogger::endl;
      return kERROR;
    }
    mcsectornamestr << "TPCDigitMCTruth" << mClusterSector;
    mDigitMCTruthArray[mClusterSector] =
      mgr->InitObjectAs<const dataformats::MCTruthContainer<MCCompLabel>*>(mcsectornamestr.str().c_str());
    if (!mDigitMCTruthArray[mClusterSector]) {
      LOG(ERROR) << "TPC MC Truth not registered in the FairRootManager. Exiting ..." << FairLogger::endl;
      return kERROR;
    }
  } else {
    // in case we are treating all sectors
    for (int s = 0; s < Sector::MAXSECTOR; ++s) {
      std::stringstream sectornamestr;
      std::stringstream mcsectornamestr;
      sectornamestr << "TPCDigit" << s;
      LOG(INFO) << "FETCHING DIGITS FOR SECTOR " << s << "\n";
      mDigitsArray[s] = mgr->InitObjectAs<const std::vector<Digit>*>(sectornamestr.str().c_str());
      if (!mDigitsArray[s]) {
        LOG(ERROR) << "TPC points not registered in the FairRootManager. Exiting ..." << FairLogger::endl;
        return kERROR;
      }
      mcsectornamestr << "TPCDigitMCTruth" << s;
      mDigitMCTruthArray[s] =
        mgr->InitObjectAs<const dataformats::MCTruthContainer<MCCompLabel>*>(mcsectornamestr.str().c_str());
      if (!mDigitMCTruthArray[s]) {
        LOG(ERROR) << "TPC MC Truth not registered in the FairRootManager. Exiting ..." << FairLogger::endl;
        return kERROR;
      }
    }
  }

    // Register output container
    mHwClustersArray = std::make_unique<std::vector<o2::TPC::Cluster>>();
    // a trick to register the unique pointer with FairRootManager
    static auto tmp = mHwClustersArray.get();
    mgr->RegisterAny("TPCClusterHW", tmp, kTRUE);

    // Register MC Truth output container
    mHwClustersMCTruthArray = std::make_unique<MCLabelContainer>();
    // a trick to register the unique pointer with FairRootManager
    static auto tmp2 = mHwClustersMCTruthArray.get();
    mgr->RegisterAny("TPCClusterHWMCTruth", tmp2, kTRUE);

     // create clusterer and pass output pointer
    mHwClusterer = std::make_unique<HwClusterer>(mHwClustersArray.get(),mHwClustersMCTruthArray.get());//,0,359);
    mHwClusterer->setContinuousReadout(mIsContinuousReadout);
// TODO: implement noise/pedestal objecta
//    mHwClusterer->setNoiseObject();
//    mHwClusterer->setPedestalObject();

  return kSUCCESS;
}

//_____________________________________________________________________
void ClustererTask::Exec(Option_t *option)
{
  LOG(DEBUG) << "Running clusterization on new event" << FairLogger::endl;

    if (mHwClustersArray)
      mHwClustersArray->clear();
    if (mHwClustersMCTruthArray)
      mHwClustersMCTruthArray->clear();
    if (mClusterSector != -1) {
      mHwClusterer->Process(*mDigitsArray[mClusterSector], mDigitMCTruthArray[mClusterSector], mEventCount);
      LOG(DEBUG) << "Hw clusterer found " << mHwClustersArray->size() << " clusters" << FairLogger::endl
                 << FairLogger::endl;
    } else {
      for (int s = 0; s < Sector::MAXSECTOR; ++s) {
        mHwClusterer->Process(*mDigitsArray[s], mDigitMCTruthArray[s], mEventCount);
        LOG(DEBUG) << "Hw clusterer found" << mHwClustersArray->size() << " clusters" << FairLogger::endl
                   << FairLogger::endl;
      }
    }
  ++mEventCount;
}
