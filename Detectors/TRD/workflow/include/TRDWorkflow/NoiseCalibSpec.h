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

#ifndef O2_TRD_NOISECALIBSPEC_H
#define O2_TRD_NOISECALIBSPEC_H

/// \file   NoiseCalibSpec.h
/// \brief Extract mean ADC values per pad from digits and send them to the aggregator

#include "Framework/Task.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/ControlService.h"
#include "Framework/WorkflowSpec.h"
#include "DataFormatsTRD/NoiseCalibration.h"
#include "DetectorsBase/TFIDInfoHelper.h"
#include "TRDCalibration/CalibratorNoise.h"

using namespace o2::framework;

namespace o2
{
namespace trd
{

class TRDNoiseCalibSpec : public o2::framework::Task
{
 public:
  TRDNoiseCalibSpec(bool dummy) : mIsDummy(dummy) {}
  void init(o2::framework::InitContext& ic) final
  {
    // Do we need to initialize something?
  }

  void run(o2::framework::ProcessingContext& pc) final
  {
    auto digits = pc.inputs().get<gsl::span<Digit>>("trddigits");
    auto trigRecs = pc.inputs().get<gsl::span<TriggerRecord>>("trdtriggerrec");

    if (mIsDummy) {
      return;
    }

    // Obtain rough time from the data header (done only once)
    if (mStartTime == 0) {
      o2::dataformats::TFIDInfo ti;
      o2::base::TFIDInfoHelper::fillTFIDInfo(pc, ti);
      if (!ti.isDummy()) {
        mStartTime = ti.creation;
      }
    }

    // process the digits, as long as we have not reached the limit
    if (!mCalibrator.hasEnoughData()) {
      mCalibrator.process(digits);
    } else {
      if (!mHaveSentOutput) {
        LOGP(important, "Enough data received after {} TFs seen, finalizing noise calibration", mNTFsProcessed);
        mCalibrator.collectChannelInfo();
        sendOutput(pc.outputs());
        mHaveSentOutput = true;
      } else {
        if ((mNTFsProcessed % 200) == 0) {
          LOGP(important, "Not processing anymore. Seen {} TFs in total. Run can be stopped", mNTFsProcessed);
        }
      }
    }
    ++mNTFsProcessed;
  }

  void sendOutput(DataAllocator& output)
  {
    // extract CCDB infos and calibration objects, convert it to TMemFile and send them to the output

    const auto& payload = mCalibrator.getCcdbObject();

    auto clName = o2::utils::MemFileHelper::getClassName(payload);
    auto flName = o2::ccdb::CcdbApi::generateFileName(clName);
    std::map<std::string, std::string> metadata; // do we want to add something?
    long startValidity = mStartTime;
    o2::ccdb::CcdbObjectInfo info("TRD/Calib/ChannelStatus", clName, flName, metadata, startValidity, startValidity + 3 * o2::ccdb::CcdbObjectInfo::MONTH);

    auto image = o2::ccdb::CcdbApi::createObjectImage(&payload, &info);
    LOG(info) << "Sending object " << info.getPath() << "/" << info.getFileName() << " of size " << image->size()
              << " bytes, valid for " << info.getStartValidityTimestamp() << " : " << info.getEndValidityTimestamp();

    output.snapshot(Output{o2::calibration::Utils::gDataOriginCDBPayload, "PADSTATUS", 0}, *image.get()); // vector<char>
    output.snapshot(Output{o2::calibration::Utils::gDataOriginCDBWrapper, "PADSTATUS", 0}, info);         // root-serialized
  }

  void endOfStream(o2::framework::EndOfStreamContext& ec) final
  {
    if (mHaveSentOutput) {
      LOGP(important, "Received EoS after sending calibration object. All OK");
    } else {
      if (!mIsDummy) {
        LOGP(alarm, "Received EoS before sending calibration object. Not enough digits received");
      }
    }
  }

 private:
  CalibratorNoise mCalibrator{};
  size_t mNTFsProcessed{0};
  bool mHaveSentOutput{false};
  bool mIsDummy{false};
  uint64_t mStartTime{0};
};

o2::framework::DataProcessorSpec getTRDNoiseCalibSpec(bool dummy)
{
  std::vector<InputSpec> inputs;
  inputs.emplace_back("trddigits", o2::header::gDataOriginTRD, "DIGITS", 0, Lifetime::Timeframe);
  inputs.emplace_back("trdtriggerrec", o2::header::gDataOriginTRD, "TRKTRGRD", 0, Lifetime::Timeframe);

  std::vector<OutputSpec> outputs;
  outputs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBPayload, "PADSTATUS"}, Lifetime::Sporadic);
  outputs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBWrapper, "PADSTATUS"}, Lifetime::Sporadic);

  return DataProcessorSpec{
    "trd-noise-calib",
    inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<TRDNoiseCalibSpec>(dummy)},
    Options{}};
}

} // namespace trd
} // namespace o2

#endif // O2_TRD_NOISECALIBSPEC_H
