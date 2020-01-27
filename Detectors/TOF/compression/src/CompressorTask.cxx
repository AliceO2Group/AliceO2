// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   CompressorTask.cxx
/// @author Roberto Preghenella
/// @since  2019-12-18
/// @brief  TOF raw data compressor task

#include "TOFCompression/CompressorTask.h"
#include "Framework/ControlService.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/WorkflowSpec.h"

using namespace o2::framework;

namespace o2
{
namespace tof
{

void CompressorTask::init(InitContext& ic)
{
  LOG(INFO) << "Compressor init";

  /** link encoder output buffer **/
  mCompressor.setEncoderBuffer(const_cast<char*>(mDataFrame.mBuffer));

  auto decoderVerbose = ic.options().get<bool>("tof-compressor-decoder-verbose");
  auto encoderVerbose = ic.options().get<bool>("tof-compressor-encoder-verbose");
  auto checkerVerbose = ic.options().get<bool>("tof-compressor-checker-verbose");

  mCompressor.setDecoderVerbose(decoderVerbose);
  mCompressor.setEncoderVerbose(encoderVerbose);
  mCompressor.setCheckerVerbose(checkerVerbose);

  auto finishFunction = [this]() {
    mCompressor.checkSummary();
  };

  ic.services().get<CallbackService>().set(CallbackService::Id::Stop, finishFunction);
}

void CompressorTask::run(ProcessingContext& pc)
{
  LOG(DEBUG) << "Compressor run";

  /** receive input **/
  for (auto& input : pc.inputs()) {
    const auto* header = DataRefUtils::getHeader<o2::header::DataHeader*>(input);
    auto payload = const_cast<char*>(input.payload);
    auto payloadSize = header->payloadSize;
    mCompressor.setDecoderBuffer(payload);
    mCompressor.setDecoderBufferSize(payloadSize);

    /** run **/
    mCompressor.run();

    /** push output **/
    mDataFrame.mSize = mCompressor.getEncoderByteCounter();
    pc.outputs().snapshot(Output{o2::header::gDataOriginTOF, "CMPDATAFRAME", 0, Lifetime::Timeframe}, mDataFrame);
  }
}

DataProcessorSpec CompressorTask::getSpec()
{
  std::vector<InputSpec> inputs;
  std::vector<OutputSpec> outputs;

  return DataProcessorSpec{
    "tof-compressor",
    select("x:TOF/RAWDATA"),
    Outputs{OutputSpec(o2::header::gDataOriginTOF, "CMPDATAFRAME", 0, Lifetime::Timeframe)},
    AlgorithmSpec{adaptFromTask<CompressorTask>()},
    Options{
      {"tof-compressor-decoder-verbose", VariantType::Bool, false, {"Decoder verbose flag"}},
      {"tof-compressor-encoder-verbose", VariantType::Bool, false, {"Encoder verbose flag"}},
      {"tof-compressor-checker-verbose", VariantType::Bool, false, {"Checker verbose flag"}}}};
}

} // namespace tof
} // namespace o2
