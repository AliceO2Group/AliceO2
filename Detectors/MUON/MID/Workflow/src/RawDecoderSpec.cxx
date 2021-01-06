// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file   MID/Workflow/src/RawDecoderSpec.cxx
/// \brief  Data processor spec for MID raw decoder device
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   26 February 2020

#include "MIDWorkflow/RawDecoderSpec.h"

#include <chrono>
#include "Framework/CallbackService.h"
#include "Framework/Logger.h"
#include "Framework/Output.h"
#include "Framework/Task.h"
#include "DPLUtils/DPLRawParser.h"
#include "Headers/RDHAny.h"
#include "MIDRaw/Decoder.h"

namespace of = o2::framework;

namespace o2
{
namespace mid
{

class RawDecoderDeviceDPL
{
 public:
  RawDecoderDeviceDPL(bool isDebugMode, const FEEIdConfig& feeIdConfig, const CrateMasks& crateMasks, const ElectronicsDelay& electronicsDelay, header::DataHeader::SubSpecificationType subSpec) : mIsDebugMode(isDebugMode), mFeeIdConfig(feeIdConfig), mCrateMasks(crateMasks), mElectronicsDelay(electronicsDelay), mSubSpec(subSpec) {}

  void init(of::InitContext& ic)
  {
    auto stop = [this]() {
      LOG(INFO) << "Capacities: ROFRecords: " << mDecoder->getROFRecords().capacity() << "  LocalBoards: " << mDecoder->getData().capacity();
      double scaleFactor = 1.e6 / mNROFs;
      LOG(INFO) << "Processing time / " << mNROFs << " ROFs: full: " << mTimer.count() * scaleFactor << " us  decoding: " << mTimerAlgo.count() * scaleFactor << " us";
    };
    ic.services().get<of::CallbackService>().set(of::CallbackService::Id::Stop, stop);
  }

  void run(of::ProcessingContext& pc)
  {
    auto tStart = std::chrono::high_resolution_clock::now();

    auto tAlgoStart = std::chrono::high_resolution_clock::now();
    of::DPLRawParser parser(pc.inputs());

    if (!mDecoder) {
      auto const* rdhPtr = reinterpret_cast<const o2::header::RDHAny*>(parser.begin().raw());
      mDecoder = createDecoder(*rdhPtr, mIsDebugMode, mElectronicsDelay, mCrateMasks, mFeeIdConfig);
    }

    mDecoder->clear();
    for (auto it = parser.begin(), end = parser.end(); it != end; ++it) {
      auto const* rdhPtr = reinterpret_cast<const o2::header::RDHAny*>(it.raw());
      gsl::span<const uint8_t> payload(it.data(), it.size());
      mDecoder->process(payload, *rdhPtr);
    }

    mTimerAlgo += std::chrono::high_resolution_clock::now() - tAlgoStart;

    pc.outputs().snapshot(of::Output{header::gDataOriginMID, "DECODED", mSubSpec}, mDecoder->getData());
    pc.outputs().snapshot(of::Output{header::gDataOriginMID, "DECODEDROF", mSubSpec}, mDecoder->getROFRecords());

    mTimer += std::chrono::high_resolution_clock::now() - tStart;
    mNROFs += mDecoder->getROFRecords().size();
  }

 private:
  std::unique_ptr<Decoder> mDecoder{nullptr};
  bool mIsDebugMode{false};
  FEEIdConfig mFeeIdConfig{};
  CrateMasks mCrateMasks{};
  ElectronicsDelay mElectronicsDelay{};
  header::DataHeader::SubSpecificationType mSubSpec{0};
  std::chrono::duration<double> mTimer{0};     ///< full timer
  std::chrono::duration<double> mTimerAlgo{0}; ///< algorithm timer
  unsigned int mNROFs{0};                      /// Total number of processed ROFs
};

of::DataProcessorSpec getRawDecoderSpec(bool isDebugMode, const FEEIdConfig& feeIdConfig, const CrateMasks& crateMasks, const ElectronicsDelay& electronicsDelay, const std::vector<of::InputSpec>& inputSpecs, o2::header::DataHeader::SubSpecificationType subSpecType)
{
  std::vector<of::OutputSpec> outputSpecs{of::OutputSpec{header::gDataOriginMID, "DECODED", subSpecType, of::Lifetime::Timeframe}, of::OutputSpec{header::gDataOriginMID, "DECODEDROF", subSpecType, of::Lifetime::Timeframe}};
  return of::DataProcessorSpec{
    "MIDRawDecoder",
    {inputSpecs},
    {outputSpecs},
    of::adaptFromTask<o2::mid::RawDecoderDeviceDPL>(isDebugMode, feeIdConfig, crateMasks, electronicsDelay, subSpecType)};
}

of::DataProcessorSpec getRawDecoderSpec(bool isDebugMode)
{
  return getRawDecoderSpec(isDebugMode, FEEIdConfig(), CrateMasks(), ElectronicsDelay());
}

of::DataProcessorSpec getRawDecoderSpec(bool isDebugMode, const FEEIdConfig& feeIdConfig, const CrateMasks& crateMasks, const ElectronicsDelay& electronicsDelay)
{
  std::vector<of::InputSpec> inputSpecs{{"mid_raw", of::ConcreteDataTypeMatcher{header::gDataOriginMID, header::gDataDescriptionRawData}, of::Lifetime::Timeframe}};
  header::DataHeader::SubSpecificationType subSpec{0};
  return getRawDecoderSpec(isDebugMode, FEEIdConfig(), CrateMasks(), ElectronicsDelay(), inputSpecs, subSpec);
}

of::DataProcessorSpec getRawDecoderSpec(bool isDebugMode, const FEEIdConfig& feeIdConfig, const CrateMasks& crateMasks, const ElectronicsDelay& electronicsDelay, header::DataHeader::SubSpecificationType subSpec)
{
  std::vector<of::InputSpec> inputSpecs{{"mid_raw", header::gDataOriginMID, header::gDataDescriptionRawData, subSpec, o2::framework::Lifetime::Timeframe}};

  return getRawDecoderSpec(isDebugMode, feeIdConfig, crateMasks, electronicsDelay, inputSpecs, subSpec);
}
} // namespace mid
} // namespace o2
