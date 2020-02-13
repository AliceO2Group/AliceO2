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
#include "Framework/RawDeviceService.h"
#include "Framework/DeviceSpec.h"
#include "Framework/DataSpecUtils.h"

#include <fairmq/FairMQDevice.h>

using namespace o2::framework;

namespace o2
{
namespace tof
{

void CompressorTask::init(InitContext& ic)
{
  LOG(INFO) << "Compressor init";

  auto decoderVerbose = ic.options().get<bool>("decoder-verbose");
  auto encoderVerbose = ic.options().get<bool>("encoder-verbose");
  auto checkerVerbose = ic.options().get<bool>("checker-verbose");

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

  /** set encoder output buffer **/
  char bufferOut[1048576];
  mCompressor.setEncoderBuffer(bufferOut);
  mCompressor.setEncoderBufferSize(1048576);

  auto device = pc.services().get<o2::framework::RawDeviceService>().device();
  auto outputRoutes = pc.services().get<o2::framework::RawDeviceService>().spec().outputs;
  auto fairMQChannel = outputRoutes.at(0).channel;

  /** receive input **/
  for (auto& input : pc.inputs()) {

    /** input **/
    auto headerIn = DataRefUtils::getHeader<o2::header::DataHeader*>(input);
    auto dataProcessingHeaderIn = DataRefUtils::getHeader<o2::framework::DataProcessingHeader*>(input);
    auto payloadIn = const_cast<char*>(input.payload);
    auto payloadInSize = headerIn->payloadSize;
    mCompressor.setDecoderBuffer(payloadIn);
    mCompressor.setDecoderBufferSize(payloadInSize);

    /** run **/
    mCompressor.run();
    auto payloadOutSize = mCompressor.getEncoderByteCounter();
    auto payloadMessage = device->NewMessage(payloadOutSize);
    std::memcpy(payloadMessage->GetData(), bufferOut, payloadOutSize);

    /** output **/
    auto headerOut = *headerIn;
    auto dataProcessingHeaderOut = *dataProcessingHeaderIn;
    headerOut.dataDescription = "CRAWDATA";
    headerOut.payloadSize = payloadOutSize;
    o2::header::Stack headerStack{headerOut, dataProcessingHeaderOut};
    auto headerMessage = device->NewMessage(headerStack.size());
    std::memcpy(headerMessage->GetData(), headerStack.data(), headerStack.size());

    /** send **/
    FairMQParts parts;
    parts.AddPart(std::move(headerMessage));
    parts.AddPart(std::move(payloadMessage));
    device->Send(parts, fairMQChannel);
  }
}

} // namespace tof
} // namespace o2
