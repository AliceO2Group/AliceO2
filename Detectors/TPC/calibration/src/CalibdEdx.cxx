// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "TPCCalibration/CalibdEdx.h"

#include <array>
#include <cstddef>
#include <memory>
#include <string_view>

//o2 includes
#include "CommonUtils/MemFileHelper.h"
#include "CommonUtils/TreeStreamRedirector.h"
#include "CCDB/CcdbApi.h"
#include "DetectorsCalibration/Utils.h"
#include "Framework/Logger.h"
#include "TPCCalibration/dEdxHistos.h"

using namespace o2::tpc;

void CalibdEdx::initOutput()
{
  // Here we initialize the vector of our output objects
  mInfoVector.clear();
  mMIPVector.clear();
  return;
}

void CalibdEdx::finalizeSlot(Slot& slot)
{
  LOG(INFO) << "Finalizing slot " << slot.getTFStart() << " <= TF <= " << slot.getTFEnd();

  const dEdxHistos* container = slot.getContainer();
  const auto statsASide = container->getHists()[0].getStatisticsData();
  const auto statsCSide = container->getHists()[1].getStatisticsData();

  slot.print();
  LOG(INFO) << "A side, truncated mean statistics: Mean = " << statsASide.mCOG << ", StdDev = " << statsCSide.mStdDev << ", Entries = " << statsASide.mSum;
  LOG(INFO) << "C side, truncated mean statistics: Mean = " << statsCSide.mCOG << ", StdDev = " << statsCSide.mStdDev << ", Entries = " << statsCSide.mSum;

  CalibMIP mips{statsASide.mCOG, statsCSide.mCOG};

  const auto className = o2::utils::MemFileHelper::getClassName(mips);
  const auto fileName = o2::ccdb::CcdbApi::generateFileName(className);
  const std::map<std::string, std::string> metaData;

  // TODO: the timestamp is now given with the TF index, but it will have
  // to become an absolute time.
  TFType timeFrame = slot.getTFStart();
  mInfoVector.emplace_back("TPC/Calib/MIPS", className, fileName, metaData, timeFrame, 99999999999999);
  mMIPVector.push_back(mips);

  if (mDebugOutputStreamer) {
    LOG(INFO) << "Dumping time slot data to file";

    *mDebugOutputStreamer << "mipPosition"
                          << "timeFrame=" << timeFrame            // Initial time frame of time slot
                          << "calibMIP=" << mips                  // Computed MIP positions
                          << "dEdxHistos=" << slot.getContainer() // dE/dx histograms
                          << "\n";
  }
}

CalibdEdx::Slot& CalibdEdx::emplaceNewSlot(bool front, TFType tstart, TFType tend)
{
  auto& cont = getSlots();
  auto& slot = front ? cont.emplace_front(tstart, tend) : cont.emplace_back(tstart, tend);

  auto container = std::make_unique<dEdxHistos>(mNBins, mCuts);
  container->setApplyCuts(mApplyCuts);

  slot.setContainer(std::move(container));
  return slot;
}

void CalibdEdx::enableDebugOutput(std::string_view fileName)
{
  mDebugOutputStreamer = std::make_unique<o2::utils::TreeStreamRedirector>(fileName.data(), "recreate");
}

void CalibdEdx::disableDebugOutput()
{
  // This will call the TreeStream destructor and write any stored data.
  mDebugOutputStreamer.reset();
}

void CalibdEdx::finalizeDebugOutput() const
{
  if (mDebugOutputStreamer) {
    LOG(INFO) << "Closing dump file";
    mDebugOutputStreamer->Close();
  }
}
