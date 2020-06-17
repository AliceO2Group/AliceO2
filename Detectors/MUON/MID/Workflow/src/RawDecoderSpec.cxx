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
#include "Framework/ConfigParamRegistry.h"
#include "Framework/ControlService.h"
#include "Framework/Logger.h"
#include "Framework/Output.h"
#include "Framework/Task.h"
#include "Framework/WorkflowSpec.h"
#include "DPLUtils/DPLRawParser.h"
#include "MIDRaw/FEEIdConfig.h"
#include "MIDRaw/CrateMasks.h"
#include "MIDRaw/Decoder.h"
#include "MIDRaw/GBTBareDecoder.h"
#include "MIDRaw/GBTUserLogicDecoder.h"

namespace of = o2::framework;

namespace o2
{
namespace mid
{

template <typename GBTDECODER>
class RawDecoderDeviceDPL
{
 public:
  void init(of::InitContext& ic)
  {
    auto stop = [this]() {
      LOG(INFO) << "Capacities: ROFRecords: " << mDecoder.getROFRecords().capacity() << "  LocalBoards: " << mDecoder.getData().capacity();
      double scaleFactor = 1.e6 / mNROFs;
      LOG(INFO) << "Processing time / " << mNROFs << " ROFs: full: " << mTimer.count() * scaleFactor << " us  decoding: " << mTimerAlgo.count() * scaleFactor << " us";
    };
    ic.services().get<of::CallbackService>().set(of::CallbackService::Id::Stop, stop);

    auto feeIdConfigFilename = ic.options().get<std::string>("feeId-config-file");
    if (!feeIdConfigFilename.empty()) {
      o2::mid::FEEIdConfig feeIdConfig(feeIdConfigFilename.c_str());
      mDecoder.setFeeIdConfig(feeIdConfig);
    }
    auto crateMasksFilename = ic.options().get<std::string>("crate-masks-file");
    if (!crateMasksFilename.empty()) {
      o2::mid::CrateMasks crateMasks(crateMasksFilename.c_str());
      mDecoder.setCrateMasks(crateMasks);
    }

    auto isDebugMode = ic.options().get<bool>("debug-mode");
    mDecoder.init(isDebugMode);
  }

  void run(of::ProcessingContext& pc)
  {
    auto tStart = std::chrono::high_resolution_clock::now();

    auto msg = pc.inputs().get("mid_raw");
    auto buffer = of::DataRefUtils::as<const uint8_t>(msg);

    auto tAlgoStart = std::chrono::high_resolution_clock::now();
    of::DPLRawParser parser(pc.inputs());

    mDecoder.clear();
    for (auto it = parser.begin(), end = parser.end(); it != end; ++it) {
      // retrieving RDH v4
      auto rdhPtr = it.get_if<o2::header::RAWDataHeader>();
      gsl::span<const uint8_t> payload(it.data(), it.size());
      mDecoder.process(payload, *rdhPtr);
    }

    mDecoder.flush();
    mTimerAlgo += std::chrono::high_resolution_clock::now() - tAlgoStart;

    pc.outputs().snapshot(of::Output{header::gDataOriginMID, "DECODED", 0, of::Lifetime::Timeframe}, mDecoder.getData());
    pc.outputs().snapshot(of::Output{header::gDataOriginMID, "DECODEDROF", 0, of::Lifetime::Timeframe}, mDecoder.getROFRecords());

    mTimer += std::chrono::high_resolution_clock::now() - tStart;
    mNROFs += mDecoder.getROFRecords().size();
  }

 private:
  Decoder<GBTDECODER> mDecoder{};
  std::chrono::duration<double> mTimer{0};     ///< full timer
  std::chrono::duration<double> mTimerAlgo{0}; ///< algorithm timer
  unsigned int mNROFs{0};                      /// Total number of processed ROFs
};

of::AlgorithmSpec getAlgorithmSpec(bool isBare)
{
  if (isBare) {
    return of::adaptFromTask<o2::mid::RawDecoderDeviceDPL<o2::mid::GBTBareDecoder>>();
  }
  return of::adaptFromTask<o2::mid::RawDecoderDeviceDPL<o2::mid::GBTUserLogicDecoder>>();
}

framework::DataProcessorSpec getRawDecoderSpec(bool isBare)
{
  std::vector<of::InputSpec> inputSpecs{of::InputSpec{"mid_raw", of::ConcreteDataTypeMatcher{header::gDataOriginMID, header::gDataDescriptionRawData}, of::Lifetime::Timeframe}};
  std::vector<of::OutputSpec> outputSpecs{of::OutputSpec{header::gDataOriginMID, "DECODED", 0, of::Lifetime::Timeframe}, of::OutputSpec{header::gDataOriginMID, "DECODEDROF", 0, of::Lifetime::Timeframe}};

  return of::DataProcessorSpec{
    "MIDRawDecoder",
    {inputSpecs},
    {outputSpecs},
    getAlgorithmSpec(isBare),
    of::Options{
      {"feeId-config-file", of::VariantType::String, "", {"Filename with crate FEE ID correspondence"}},
      {"crate-masks-file", of::VariantType::String, "", {"Filename with crate masks"}},
      {"debug-mode", of::VariantType::Bool, false, {"Debug mode: sends all boards"}}}};
}
} // namespace mid
} // namespace o2
