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
#include "DataFormatsTOF/CalibInfoTOF.h"
#include "TOFBase/Geo.h"
#include "CommonConstants/MathConstants.h"
#include "DataFormatsTOF/Diagnostic.h"
#include "DetectorsRaw/HBFUtils.h"

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
    mMinSize = ic.options().get<int>("min-number-of-info");
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
    int size = mMinSize + gRandom->Integer(100); // push dummy output
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
  int mMinSize = 100;
};

class TFProcessorCalibInfoTOF : public o2::framework::Task
{
 public:
  TFProcessorCalibInfoTOF(int latency, int latencyRMS) : mMeanLatency(latency), mLatencyRMS(latencyRMS){};

  void init(o2::framework::InitContext& ic) final
  {
    mDevCopy = ic.services().get<const o2::framework::DeviceSpec>().inputTimesliceId;
    gRandom->SetSeed(mDevCopy);
    mTOFChannelCalib = ic.options().get<bool>("do-TOF-channel-calib");
    mTOFChannelCalibInTestMode = ic.options().get<bool>("do-TOF-channel-calib-in-test-mode");
    LOG(info) << "TFProcessorCalibInfoTOFCopy: " << mDevCopy << " MeanLatency: " << mMeanLatency << " LatencyRMS: " << mLatencyRMS << " DoTOFChannelCalib: " << mTOFChannelCalib
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
    LOG(info) << "TFProcessorCalibInfoTOFCopy: " << mDevCopy << " Simulate latency of " << delay << " mcs for TF " << tfcounter;
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

class TFProcessorDiagnostic : public o2::framework::Task
{
 public:
  TFProcessorDiagnostic(int latency, int latencyRMS) : mMeanLatency(latency), mLatencyRMS(latencyRMS){};

  void init(o2::framework::InitContext& ic) final
  {

    mDevCopy = ic.services().get<const o2::framework::DeviceSpec>().inputTimesliceId;
    gRandom->SetSeed(mDevCopy);
    LOG(info) << "TFProcessorDiagnosticCopy: " << mDevCopy;

    auto size = ic.options().get<int>("n-diag-words"); // number of diagnostic words that we want to simulate; only these will then be present
    mProb.resize(size);
    mDiagnosticPattern.resize(size);

    for (int i = 0; i < size; ++i) {
      int crateId = gRandom->Integer(72);
      int trmId = gRandom->Integer(10);
      mProb[i] = 0;
      mDiagnosticPattern[i] = (ULong64_t(trmId) << 32) + (ULong64_t(crateId) << 36);
      for (int j = 0; j < 28; ++j) { // adding diagnostic patterns for each word; we can have 28 at max, but we don't want all of them (we use a cut on a probability of 30%)
        if (gRandom->Rndm() < 0.3) {
          mDiagnosticPattern[i] += 1 << j;
          mProb[i] = gRandom->Rndm(); // probability for each diagnostic word
        }
      }
    }
  }

  void run(o2::framework::ProcessingContext& pc) final
  {
    auto tfcounter = o2::header::get<o2::framework::DataProcessingHeader*>(pc.inputs().get("input").header)->startTime;
    // introduceDelay
    uint32_t delay = std::abs(gRandom->Gaus(mMeanLatency, mLatencyRMS));
    LOG(info) << "TFProcessorDiagnosticCopy: " << mDevCopy << " Simulate latency of " << delay << " mcs for TF " << tfcounter;
    usleep(delay);

    // push dummy output
    auto& outputDiagnostic = pc.outputs().make<o2::tof::Diagnostic>(o2::framework::OutputRef{"output", 0});
    for (int iOrbit = 0; iOrbit < o2::raw::HBFUtils::Instance().getNOrbitsPerTF(); ++iOrbit) {
      for (int iROwindow = 0; iROwindow < o2::tof::Geo::NWINDOW_IN_ORBIT; ++iROwindow) {
        outputDiagnostic.fillROW();
        for (int i = 0; i < mDiagnosticPattern.size(); ++i) {
          if (gRandom->Rndm() < mProb[i]) {
            outputDiagnostic.fill(mDiagnosticPattern[i]);
          }
        }
      }
    }
    LOG(debug) << "diagnostic for TF " << tfcounter << " --> ";
    outputDiagnostic.print();
  }

 private:
  int mDevCopy = 0;
  std::vector<uint64_t> mDiagnosticPattern;
  std::vector<float> mProb;
  o2::tof::Diagnostic mDiagnosticFrequency;
  uint32_t mMeanLatency = 0;
  uint32_t mLatencyRMS = 1;
};

} // namespace calibration

namespace framework
{

DataProcessorSpec getTFDispatcherSpec(int slot, int ngen, int nlanes, int latency)
{
  return DataProcessorSpec{
    "calib-tf-dispatcher",
    Inputs{},
    Outputs{{{"output"}, "TOF", "DATASIZE"}},
    AlgorithmSpec{adaptFromTask<o2::calibration::TFDispatcher>(slot, ngen, nlanes, latency)},
    Options{{"max-timeframes", VariantType::Int64, 99999999999ll, {"max TimeFrames to generate"}},
            {"min-number-of-info", VariantType::Int, 9999, {"min number of Info (CalibTOFInfo, or Diagnostic) to generate"}}}};
}

DataProcessorSpec getTFProcessorCalibInfoTOFSpec(int latency, int latencyRMS)
{

  return DataProcessorSpec{
    "calib-tf-data-processor",
    Inputs{{"input", "TOF", "DATASIZE"}},
    Outputs{{{"output"}, "TOF", "CALIBDATA"}},
    AlgorithmSpec{adaptFromTask<o2::calibration::TFProcessorCalibInfoTOF>(latency, latencyRMS)},
    Options{
      {"do-TOF-channel-calib", VariantType::Bool, false, {"flag to do TOF ChannelCalib"}},
      {"do-TOF-channel-calib-in-test-mode", VariantType::Bool, false, {"flag to do TOF ChannelCalib in testMode"}}}};
}

DataProcessorSpec getTFProcessorDiagnosticSpec(int latency, int latencyRMS)
{

  return DataProcessorSpec{
    "calib-tf-data-processor",
    Inputs{{"input", "TOF", "DATASIZE"}},
    Outputs{{{"output"}, "TOF", "DIAFREQ"}},
    AlgorithmSpec{adaptFromTask<o2::calibration::TFProcessorDiagnostic>(latency, latencyRMS)},
    Options{
      {"n-diag-words", VariantType::Int, 20, {"number of diagnostic words to simulate"}}}};
}

} // namespace framework
} // namespace o2

#endif
