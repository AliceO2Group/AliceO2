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

#include "TPCCalibration/CalibratordEdx.h"

#include <array>
#include <cstddef>
#include <memory>
#include <string_view>

// o2 includes
#include "CommonUtils/MemFileHelper.h"
#include "CommonUtils/TreeStreamRedirector.h"
#include "CCDB/CcdbApi.h"
#include "DetectorsCalibration/Utils.h"
#include "Framework/Logger.h"
#include "TPCCalibration/CalibdEdx.h"

using namespace o2::tpc;

void CalibratordEdx::initOutput()
{
  // Here we initialize the vector of our output objects
  mInfoVector.clear();
  mCalibVector.clear();
}

void CalibratordEdx::finalizeSlot(Slot& slot)
{
  LOGP(info, "Finalizing slot {} <= TF <= {}", slot.getTFStart(), slot.getTFEnd());

  // compute calibration values from histograms
  CalibdEdx* container = slot.getContainer();
  container->finalize();
  const auto& calib = container->getCalib();

  slot.print();

  const auto className = o2::utils::MemFileHelper::getClassName(calib);
  const auto fileName = o2::ccdb::CcdbApi::generateFileName(className);
  const std::map<std::string, std::string> metaData;

  // TODO: the timestamp is now given with the TF index, but it will have
  // to become an absolute time.
  TFType startTF = slot.getTFStart();
  TFType endTF = slot.getTFEnd();
  mInfoVector.emplace_back("TPC/Calib/dEdx", className, fileName, metaData, startTF, 99999999999999);
  mCalibVector.push_back(calib);

  if (mDebugOutputStreamer) {
    LOGP(info, "Dumping time slot data to file");
    auto nonConstCalib = calib;
    *mDebugOutputStreamer << "CalibdEdx"
                          << "startTF=" << startTF          // Initial time frame of time slot
                          << "endTF=" << endTF              // Final time frame of time slot
                          << "correction=" << nonConstCalib // dE/dx corretion
                          << "\n";
  }
}

CalibratordEdx::Slot& CalibratordEdx::emplaceNewSlot(bool front, TFType tstart, TFType tend)
{
  auto& cont = getSlots();
  auto& slot = front ? cont.emplace_front(tstart, tend) : cont.emplace_back(tstart, tend);

  auto container = std::make_unique<CalibdEdx>(mMindEdx, mMaxdEdx, mdEdxBins, mZBins, mAngularBins);
  container->setApplyCuts(mApplyCuts);
  container->setCuts(mCuts);
  container->setFitCuts(mFitCuts);
  container->setField(mField);

  slot.setContainer(std::move(container));
  return slot;
}

void CalibratordEdx::enableDebugOutput(std::string_view fileName)
{
  mDebugOutputStreamer = std::make_unique<o2::utils::TreeStreamRedirector>(fileName.data(), "recreate");
}

void CalibratordEdx::disableDebugOutput()
{
  // This will call the TreeStream destructor and write any stored data.
  mDebugOutputStreamer.reset();
}

void CalibratordEdx::finalizeDebugOutput() const
{
  if (mDebugOutputStreamer) {
    LOGP(info, "Closing dump file");
    mDebugOutputStreamer->Close();
  }
}
