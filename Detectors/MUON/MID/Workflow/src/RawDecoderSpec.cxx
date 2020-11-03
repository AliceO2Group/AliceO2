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
#include "Framework/WorkflowSpec.h"
#include "DPLUtils/DPLRawParser.h"
#include "Headers/RDHAny.h"
#include "MIDRaw/Decoder.h"

namespace of = o2::framework;

namespace o2
{
namespace mid
{

template <typename GBTDECODER>
class RawDecoderDeviceDPL
{
 public:
  RawDecoderDeviceDPL<GBTDECODER>(bool isDebugMode, const FEEIdConfig& feeIdConfig, const CrateMasks& crateMasks, const ElectronicsDelay& electronicsDelay) : mIsDebugMode(isDebugMode), mFeeIdConfig(feeIdConfig), mCrateMasks(crateMasks), mElectronicsDelay(electronicsDelay) {}

  void init(of::InitContext& ic)
  {
    auto stop = [this]() {
      LOG(INFO) << "Capacities: ROFRecords: " << mDecoder.getROFRecords().capacity() << "  LocalBoards: " << mDecoder.getData().capacity();
      double scaleFactor = 1.e6 / mNROFs;
      LOG(INFO) << "Processing time / " << mNROFs << " ROFs: full: " << mTimer.count() * scaleFactor << " us  decoding: " << mTimerAlgo.count() * scaleFactor << " us";
    };
    ic.services().get<of::CallbackService>().set(of::CallbackService::Id::Stop, stop);

    mDecoder.setFeeIdConfig(mFeeIdConfig);
    mDecoder.setCrateMasks(mCrateMasks);

    mDecoder.init(mIsDebugMode);
  }

  void run(of::ProcessingContext& pc)
  {
    auto tStart = std::chrono::high_resolution_clock::now();

    auto tAlgoStart = std::chrono::high_resolution_clock::now();
    of::DPLRawParser parser(pc.inputs(), of::select("mid_raw:MID/RAWDATA"));

    mDecoder.clear();
    for (auto it = parser.begin(), end = parser.end(); it != end; ++it) {
      auto const* rdhPtr = reinterpret_cast<const o2::header::RDHAny*>(it.raw());
      gsl::span<const uint8_t> payload(it.data(), it.size());
      mDecoder.process(payload, *rdhPtr);
    }

    mDecoder.flush();
    mTimerAlgo += std::chrono::high_resolution_clock::now() - tAlgoStart;

    pc.outputs().snapshot(of::Output{header::gDataOriginMID, "DECODED", 0}, mDecoder.getData());
    pc.outputs().snapshot(of::Output{header::gDataOriginMID, "DECODEDROF", 0}, mDecoder.getROFRecords());

    mTimer += std::chrono::high_resolution_clock::now() - tStart;
    mNROFs += mDecoder.getROFRecords().size();
  }

 private:
  Decoder<GBTDECODER> mDecoder{};
  bool mIsDebugMode{false};
  FEEIdConfig mFeeIdConfig{};
  CrateMasks mCrateMasks{};
  ElectronicsDelay mElectronicsDelay{};
  std::chrono::duration<double> mTimer{0};     ///< full timer
  std::chrono::duration<double> mTimerAlgo{0}; ///< algorithm timer
  unsigned int mNROFs{0};                      /// Total number of processed ROFs
};

of::DataProcessorSpec getRawDecoderSpec(bool isBare)
{
  return getRawDecoderSpec(isBare, false, FEEIdConfig(), CrateMasks(), ElectronicsDelay());
}

of::DataProcessorSpec getRawDecoderSpec(bool isBare, bool isDebugMode, const FEEIdConfig& feeIdConfig, const CrateMasks& crateMasks, const ElectronicsDelay& electronicsDelay)
{
  std::vector<of::InputSpec> inputSpecs{of::InputSpec{"mid_raw", of::ConcreteDataTypeMatcher{header::gDataOriginMID, header::gDataDescriptionRawData}, of::Lifetime::Timeframe}};
  std::vector<of::OutputSpec> outputSpecs{of::OutputSpec{header::gDataOriginMID, "DECODED", 0, of::Lifetime::Timeframe}, of::OutputSpec{header::gDataOriginMID, "DECODEDROF", 0, of::Lifetime::Timeframe}};

  return of::DataProcessorSpec{
    "MIDRawDecoder",
    {inputSpecs},
    {outputSpecs},
    isBare ? of::adaptFromTask<o2::mid::RawDecoderDeviceDPL<o2::mid::GBTBareDecoder>>(isDebugMode, feeIdConfig, crateMasks, electronicsDelay) : of::adaptFromTask<o2::mid::RawDecoderDeviceDPL<o2::mid::GBTUserLogicDecoder>>(isDebugMode, feeIdConfig, crateMasks, electronicsDelay)};
}
} // namespace mid
} // namespace o2
