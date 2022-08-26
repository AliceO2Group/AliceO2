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

/// \file   MID/Workflow/src/RawGBTDecoderSpec.cxx
/// \brief  Data processor spec for MID GBT raw decoder device
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   06 April 2020

#include "MIDWorkflow/RawGBTDecoderSpec.h"

#include <chrono>
#include <vector>
#include "DPLUtils/DPLRawParser.h"
#include "DetectorsRaw/RDHUtils.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/ControlService.h"
#include "Framework/DataSpecUtils.h"
#include "Framework/Logger.h"
#include "Framework/Output.h"
#include "Framework/ParallelContext.h"
#include "Framework/Task.h"
#include "Headers/RDHAny.h"
#include "DetectorsRaw/RDHUtils.h"
#include "DataFormatsMID/ROBoard.h"
#include "DataFormatsMID/ROFRecord.h"
#include "MIDRaw/LinkDecoder.h"

namespace o2
{
namespace mid
{

class RawGBTDecoderDeviceDPL
{
 public:
  RawGBTDecoderDeviceDPL(bool isDebugMode, const std::vector<uint16_t>& feeIds, const CrateMasks& crateMasks, const ElectronicsDelay& electronicsDelay) : mIsDebugMode(isDebugMode), mFeeIds(feeIds), mCrateMasks(crateMasks), mElectronicsDelay(electronicsDelay) {}

  void init(o2::framework::InitContext& ic)
  {
    auto stop = [this]() {
      double scaleFactor = (mNROFs == 0) ? 0. : 1.e6 / mNROFs;
      LOG(info) << "Processing time / " << mNROFs << " ROFs: full: " << mTimer.count() * scaleFactor << " us  decoding: " << mTimerAlgo.count() * scaleFactor << " us";
    };
    ic.services().get<o2::framework::CallbackService>().set(o2::framework::CallbackService::Id::Stop, stop);

    auto idx = ic.services().get<o2::framework::ParallelContext>().index1D();
    mFeeId = mFeeIds[idx];
  }

  void run(o2::framework::ProcessingContext& pc)
  {
    auto tStart = std::chrono::high_resolution_clock::now();

    o2::framework::DPLRawParser parser(pc.inputs());

    auto tAlgoStart = std::chrono::high_resolution_clock::now();

    o2::header::DataHeader const* dh = nullptr;

    if (!mDecoder) {
      auto const* rdhPtr = reinterpret_cast<const o2::header::RDHAny*>(parser.begin().raw());
      mDecoder = createGBTDecoder(*rdhPtr, mFeeId, mIsDebugMode, mCrateMasks.getMask(mFeeId), mElectronicsDelay);
    }

    std::vector<ROBoard> data;
    std::vector<ROFRecord> rofRecords;

    for (auto it = parser.begin(), end = parser.end(); it != end; ++it) {
      dh = it.o2DataHeader();
      auto const* rdhPtr = reinterpret_cast<const o2::header::RDHAny*>(it.raw());
      gsl::span<const uint8_t> payload(it.data(), it.size());
      mDecoder->process(payload, o2::raw::RDHUtils::getHeartBeatOrbit(rdhPtr), o2::raw::RDHUtils::getTriggerType(rdhPtr), data, rofRecords);
    }

    mTimerAlgo += std::chrono::high_resolution_clock::now() - tAlgoStart;

    pc.outputs().snapshot(o2::framework::Output{header::gDataOriginMID, "DECODED", dh->subSpecification, o2::framework::Lifetime::Timeframe}, data);
    pc.outputs().snapshot(o2::framework::Output{header::gDataOriginMID, "DECODEDROF", dh->subSpecification, o2::framework::Lifetime::Timeframe}, rofRecords);

    mTimer += std::chrono::high_resolution_clock::now() - tStart;
    mNROFs += rofRecords.size();
  }

 private:
  std::unique_ptr<LinkDecoder> mDecoder{nullptr};
  bool mIsDebugMode{false};
  std::vector<uint16_t> mFeeIds{};
  CrateMasks mCrateMasks{};
  ElectronicsDelay mElectronicsDelay{};
  uint16_t mFeeId{0};
  std::chrono::duration<double> mTimer{0};     ///< full timer
  std::chrono::duration<double> mTimerAlgo{0}; ///< algorithm timer
  unsigned int mNROFs{0};                      /// Total number of processed ROFs
};

framework::DataProcessorSpec getRawGBTDecoderSpec(bool isDebugMode, const std::vector<uint16_t>& feeIds, const CrateMasks& crateMasks, const ElectronicsDelay& electronicsDelay)
{
  std::vector<o2::framework::InputSpec> inputSpecs{o2::framework::InputSpec{"mid_raw", header::gDataOriginMID, header::gDataDescriptionRawData, 0, o2::framework::Lifetime::Timeframe}};
  std::vector<o2::framework::OutputSpec> outputSpecs{o2::framework::OutputSpec{header::gDataOriginMID, "DECODED", 0, o2::framework::Lifetime::Timeframe}, o2::framework::OutputSpec{header::gDataOriginMID, "DECODEDROF", 0, o2::framework::Lifetime::Timeframe}};

  return o2::framework::DataProcessorSpec{
    "MIDRawGBTDecoder",
    {inputSpecs},
    {outputSpecs},
    o2::framework::adaptFromTask<RawGBTDecoderDeviceDPL>(isDebugMode, feeIds, crateMasks, electronicsDelay)};
}

} // namespace mid
} // namespace o2
