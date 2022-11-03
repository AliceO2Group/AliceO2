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
#include "CommonUtils/TreeStreamRedirector.h"
#include "DetectorsCalibration/Utils.h"
#include "Framework/Logger.h"
#include "TPCCalibration/CalibdEdx.h"

using namespace o2::tpc;

void CalibratordEdx::initOutput()
{
  // Here we initialize the vector of our output objects
  mTFIntervals.clear();
  mTimeIntervals.clear();
  mCalibs.clear();
}

void CalibratordEdx::finalizeSlot(Slot& slot)
{
  LOGP(info, "Finalizing slot {} <= TF <= {}", slot.getTFStart(), slot.getTFEnd());
  slot.print();

  // compute calibration values from histograms
  CalibdEdx* container = slot.getContainer();
  container->finalize();
  mCalibs.push_back(container->getCalib());

  TFType startTF = slot.getTFStart();
  TFType endTF = slot.getTFEnd();
  auto startTime = slot.getStartTimeMS();
  auto endTime = slot.getEndTimeMS();

  mTFIntervals.emplace_back(startTF, endTF);
  mTimeIntervals.emplace_back(startTime, endTime);

  if (mDebugOutputStreamer) {
    LOGP(info, "Dumping time slot data to file");
    auto calibCopy = container->getCalib();
    *mDebugOutputStreamer << "CalibdEdx"
                          << "startTF=" << startTF      // Initial time frame ID of time slot
                          << "endTF=" << endTF          // Final time frame ID of time slot
                          << "startTime=" << startTime  // Initial time frame time of time slot
                          << "endTime=" << endTime      // Final time frame time of time slot
                          << "correction=" << calibCopy // dE/dx corretion
                          << "\n";
  }
}

CalibratordEdx::Slot& CalibratordEdx::emplaceNewSlot(bool front, TFType tstart, TFType tend)
{
  auto& cont = getSlots();
  auto& slot = front ? cont.emplace_front(tstart, tend) : cont.emplace_back(tstart, tend);

  auto container = std::make_unique<CalibdEdx>(mdEdxBins, mMindEdx, mMaxdEdx, mAngularBins, mFitSnp);
  container->setApplyCuts(mApplyCuts);
  container->setCuts(mCuts);
  container->setSectorFitThreshold(mFitThreshold[0]);
  container->set1DFitThreshold(mFitThreshold[1]);
  container->set2DFitThreshold(mFitThreshold[2]);
  container->setElectronCut(mElectronCut.first, mElectronCut.second);

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
