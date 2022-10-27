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

#ifndef O2_CALIBRATION_DATAGENERATOR_H
#define O2_CALIBRATION_DATAGENERATOR_H

/// @file   DataGeneratorSpec.h
/// @brief  Dummy data generator

#include <unistd.h>
#include <TRandom.h>
#include "Framework/DeviceSpec.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/ControlService.h"
#include "Framework/WorkflowSpec.h"
#include "Framework/Task.h"
#include "Framework/Logger.h"
#include "DataFormatsZDC/CalibInfoZDC.h"
#include "ZDCBase/Constants.h"
#include "CommonConstants/MathConstants.h"

namespace o2
{
namespace calibration
{

class TFDispatcher : public o2::framework::Task
{
 public:
  TFDispatcher(int slot, int ngen, int nlanes, int latency) : mSlot(slot), mNGen(ngen), mNLanes(nlanes), mLatency(latency) {}

  void init(o2::framework::InitContext& ic) final
  {
    mMaxTF = ic.options().get<int64_t>("max-timeframes");
  }

  void run(o2::framework::ProcessingContext& pc) final
  {
    for (auto& input : pc.inputs()) {
      auto tfid = header::get<o2::framework::DataProcessingHeader*>(input.header)->startTime;
      if (tfid >= mMaxTF - 1) {
        LOG(info) << "Data generator reached TF " << tfid << ", stopping";
        pc.services().get<o2::framework::ControlService>().endOfStream();
        pc.services().get<o2::framework::ControlService>().readyToQuit(o2::framework::QuitRequest::Me);
        if (!acceptTF(tfid)) {
          return;
        }
        break;
      }
      if (!acceptTF(tfid)) {
        return;
      }
    }
    int size = 100 + gRandom->Integer(100); // push dummy output
    usleep(mLatency);
    pc.outputs().snapshot(o2::framework::OutputRef{"output", 0}, size);
  }

  bool acceptTF(int tfid)
  {

    // check if the current TF should be processed by this instance of the generator

    int targetSlot = (tfid / mNLanes) % mNGen;
    if (targetSlot != mSlot) {
      LOG(info) << "tfid = " << tfid << ", mNLanes = " << mNLanes << ", mNGen = " << mNGen << ", mSlot = " << mSlot << " target slot = " << targetSlot << ": discarded";
      return false;
    }
    LOG(info) << "tfid = " << tfid << ", mNLanes = " << mNLanes << ", mNGen = " << mNGen << ", mSlot = " << mSlot << " target slot = " << targetSlot << ": accepted";
    return true;
  }

 private:
  uint64_t mMaxTF = 1;
  int mSlot = 1;
  int mNGen = 1;
  int mNLanes = 1;
  int mLatency = 0;
};

class TFProcessor : public o2::framework::Task
{
 public:
  TFProcessor(int latency, int latencyRMS) : mMeanLatency(latency), mLatencyRMS(latencyRMS){};

  void init(o2::framework::InitContext& ic) final
  {
    mDevCopy = ic.services().get<const o2::framework::DeviceSpec>().inputTimesliceId;
    gRandom->SetSeed(mDevCopy);
    mZDCChannelCalib = ic.options().get<bool>("do-ZDC-channel-calib");
    mZDCChannelCalibInTestMode = ic.options().get<bool>("do-ZDC-channel-calib-in-test-mode");
    LOG(info) << "TFProcessorCopy: " << mDevCopy << " MeanLatency: " << mMeanLatency << " LatencyRMS: " << mLatencyRMS << " DoZDCChannelCalib: " << mZDCChannelCalib
              << " DoZDCChannelCalibInTestMode: " << mZDCChannelCalibInTestMode;
  }

  void run(o2::framework::ProcessingContext& pc) final
  {
    auto tfcounter = o2::header::get<o2::framework::DataProcessingHeader*>(pc.inputs().get("input").header)->startTime;
    // introduceDelay
    uint32_t delay = std::abs(gRandom->Gaus(mMeanLatency, mLatencyRMS));
    LOG(info) << "TFProcessorCopy: " << mDevCopy << " Simulate latency of " << delay << " mcs for TF " << tfcounter;
    usleep(delay);

    // push dummy output
    auto size = pc.inputs().get<int>("input");
    auto& output = pc.outputs().make<std::vector<o2::dataformats::CalibInfoZDC>>(o2::framework::OutputRef{"output", 0});
    output.reserve(size);

    for (int i = size; i--;) {
      if (!mZDCChannelCalib) {
        output.emplace_back(gRandom->Integer(32), 0, gRandom->Gaus(clockShift, 100.), 0, 0);
      } else {
        int channel = mZDCChannelCalibInTestMode ? gRandom->Integer(100) : gRandom->Integer(32);
        double tot = gRandom->Gaus(12, 2); // in ns
        output.emplace_back(channel, 0, value, tot, 0);
      }
    }
  }

 private:
  int mDevCopy = 0;
  uint32_t mMeanLatency = 0;
  uint32_t mLatencyRMS = 1;
  bool mZDCChannelCalib = false;
  bool mZDCChannelCalibInTestMode = false;
};

} // namespace calibration

namespace framework
{

DataProcessorSpec getTFDispatcherSpec(int slot, int ngen, int nlanes, int latency)
{
  return DataProcessorSpec{
    "calib-tf-dispatcher",
    Inputs{},
    Outputs{{{"output"}, "ZDC", "DATASIZE"}},
    AlgorithmSpec{adaptFromTask<o2::calibration::TFDispatcher>(slot, ngen, nlanes, latency)},
    Options{{"max-timeframes", VariantType::Int64, 99999999999ll, {"max TimeFrames to generate"}}}};
}

DataProcessorSpec getTFProcessorSpec(int latency, int latencyRMS)
{
  return DataProcessorSpec{
    "calib-tf-data-processor",
    Inputs{{"input", "ZDC", "DATASIZE"}},
    Outputs{{{"output"}, "ZDC", "CALIBDATA"}},
    AlgorithmSpec{adaptFromTask<o2::calibration::TFProcessor>(latency, latencyRMS)},
    Options{
      {"do-ZDC-channel-calib", VariantType::Bool, false, {"flag to do ZDC ChannelCalib"}},
      {"do-ZDC-channel-calib-in-test-mode", VariantType::Bool, false, {"flag to do ZDC ChannelCalib in testMode"}}}};
}

} // namespace framework
} // namespace o2

#endif
