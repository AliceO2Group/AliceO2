// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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
#include "DataFormatsTOF/CalibInfoTOF.h"
#include "TOFBase/Geo.h"
#include "CommonConstants/MathConstants.h"

namespace o2
{
namespace calibration
{

class TFDispatcher : public o2::framework::Task
{
 public:
  void init(o2::framework::InitContext& ic) final
  {
    mMaxTF = ic.options().get<int64_t>("max-timeframes");
  }

  void run(o2::framework::ProcessingContext& pc) final
  {
    for (auto& input : pc.inputs()) {
      auto tfid = header::get<o2::framework::DataProcessingHeader*>(input.header)->startTime;
      if (tfid >= mMaxTF) {
        LOG(INFO) << "Data generator reached TF " << tfid << ", stopping";
        pc.services().get<o2::framework::ControlService>().endOfStream();
        pc.services().get<o2::framework::ControlService>().readyToQuit(o2::framework::QuitRequest::Me);
        break;
      }
    }
    int size = 100 + gRandom->Integer(100); // push dummy output
    pc.outputs().snapshot(o2::framework::OutputRef{"output", 0}, size);
  }

 private:
  uint64_t mMaxTF = 1;
};

class TFProcessor : public o2::framework::Task
{
 public:
  void init(o2::framework::InitContext& ic) final
  {
    mDevCopy = ic.services().get<const o2::framework::DeviceSpec>().inputTimesliceId;
    gRandom->SetSeed(mDevCopy);
    mMeanLatency = std::max(1, ic.options().get<int>("mean-latency"));
    mLatencyRMS = std::max(1, ic.options().get<int>("latency-spread"));
    mTOFChannelCalib = ic.options().get<bool>("do-TOF-channel-calib");
    mTOFChannelCalibInTestMode = ic.options().get<bool>("do-TOF-channel-calib-in-test-mode");
    LOG(INFO) << "TFProcessorCopy: " << mDevCopy << " MeanLatency: " << mMeanLatency << " LatencyRMS: " << mLatencyRMS << " DoTOFChannelCalib: " << mTOFChannelCalib
              << " DoTOFChannelCalibInTestMode: " << mTOFChannelCalibInTestMode;

    for (int i = 0; i < o2::tof::Geo::NCHANNELS; i++) {
      mChannelShifts[i] = (2000. / o2::tof::Geo::NCHANNELS) * i + (-1000.); // shift needs to be always the same for a channel; in this way we make them all in [-1000, 1000], and shifting with the channel index
    }
  }

  void run(o2::framework::ProcessingContext& pc) final
  {
    auto tfcounter = o2::header::get<o2::framework::DataProcessingHeader*>(pc.inputs().get("input").header)->startTime;
    // introduceDelay
    uint32_t delay = std::abs(gRandom->Gaus(mMeanLatency, mLatencyRMS));
    LOG(INFO) << "TFProcessorCopy: " << mDevCopy << " Simulate latency of " << delay << " mcs for TF " << tfcounter;
    usleep(delay);

    // push dummy output
    auto size = pc.inputs().get<int>("input");
    auto& output = pc.outputs().make<std::vector<o2::dataformats::CalibInfoTOF>>(o2::framework::OutputRef{"output", 0});
    output.reserve(size);

    double clockShift = 1e3 * std::sin(tfcounter / 100. * o2::constants::math::PI); // in ps

    for (int i = size; i--;) {
      if (!mTOFChannelCalib) {
        output.emplace_back(gRandom->Integer(o2::tof::Geo::NCHANNELS), 0, gRandom->Gaus(clockShift, 100.), 0, 0);
      } else {
        int channel = mTOFChannelCalibInTestMode ? gRandom->Integer(100) : gRandom->Integer(o2::tof::Geo::NCHANNELS);
        double value = gRandom->Gaus(mChannelShifts[channel], 100.); // in ps
        double tot = gRandom->Gaus(12, 2);                           // in ns
        output.emplace_back(channel, 0, value, tot, 0);
      }
    }
  }

 private:
  int mDevCopy = 0;
  uint32_t mMeanLatency = 0;
  uint32_t mLatencyRMS = 1;
  bool mTOFChannelCalib = false;
  bool mTOFChannelCalibInTestMode = false;
  double mChannelShifts[o2::tof::Geo::NCHANNELS];
};

} // namespace calibration

namespace framework
{

DataProcessorSpec getTFDispatcherSpec()
{
  return DataProcessorSpec{
    "calib-tf-dispatcher",
    Inputs{},
    Outputs{{{"output"}, "DUM", "DATASIZE"}},
    AlgorithmSpec{adaptFromTask<o2::calibration::TFDispatcher>()},
    Options{{"max-timeframes", VariantType::Int64, 99999999999ll, {"max TimeFrames to generate"}}}};
}

DataProcessorSpec getTFProcessorSpec()
{
  return DataProcessorSpec{
    "calib-tf-data-processor",
    Inputs{{"input", "DUM", "DATASIZE"}},
    Outputs{{{"output"}, "DUM", "CALIBDATA"}},
    AlgorithmSpec{adaptFromTask<o2::calibration::TFProcessor>()},
    Options{
      {"mean-latency", VariantType::Int, 1000, {"mean latency of the generator in microseconds"}},
      {"latency-spread", VariantType::Int, 100, {"latency gaussian RMS of the generator in microseconds"}},
      {"do-TOF-channel-calib", VariantType::Bool, false, {"flag to do TOF ChannelCalib"}},
      {"do-TOF-channel-calib-in-test-mode", VariantType::Bool, false, {"flag to do TOF ChannelCalib in testMode"}}}};
}

} // namespace framework
} // namespace o2

#endif
