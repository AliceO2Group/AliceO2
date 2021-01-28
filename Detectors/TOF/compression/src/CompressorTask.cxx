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

template <typename RDH, bool verbose>
void CompressorTask<RDH, verbose>::init(InitContext& ic)
{
  LOG(INFO) << "Compressor init";

  auto decoderCONET = ic.options().get<bool>("tof-compressor-conet-mode");
  auto decoderVerbose = ic.options().get<bool>("tof-compressor-decoder-verbose");
  auto encoderVerbose = ic.options().get<bool>("tof-compressor-encoder-verbose");
  auto checkerVerbose = ic.options().get<bool>("tof-compressor-checker-verbose");
  mOutputBufferSize = ic.options().get<bool>("tof-compressor-output-buffer-size");

  mCompressor.setDecoderCONET(decoderCONET);
  mCompressor.setDecoderVerbose(decoderVerbose);
  mCompressor.setEncoderVerbose(encoderVerbose);
  mCompressor.setCheckerVerbose(checkerVerbose);

  auto finishFunction = [this]() {
    mCompressor.checkSummary();
  };

  ic.services().get<CallbackService>().set(CallbackService::Id::Stop, finishFunction);
}

template <typename RDH, bool verbose>
void CompressorTask<RDH, verbose>::run(ProcessingContext& pc)
{
  LOG(DEBUG) << "Compressor run";

  auto device = pc.services().get<o2::framework::RawDeviceService>().device();
  auto outputRoutes = pc.services().get<o2::framework::RawDeviceService>().spec().outputs;
  auto fairMQChannel = outputRoutes.at(0).channel;
  FairMQParts partsOut;

  /** to store data sorted by subspec id **/
  std::map<int, std::vector<o2::framework::DataRef>> subspecPartMap;
  std::map<int, int> subspecBufferSize;

  /** loop over inputs routes **/
  for (auto iit = pc.inputs().begin(), iend = pc.inputs().end(); iit != iend; ++iit) {
    if (!iit.isValid()) {
      continue;
    }

    /** loop over input parts **/
    for (auto const& ref : iit) {

      /** store parts in map **/
      auto headerIn = DataRefUtils::getHeader<o2::header::DataHeader*>(ref);
      auto subspec = headerIn->subSpecification;
      subspecPartMap[subspec].push_back(ref);

      /** increase subspec buffer size **/
      if (!subspecBufferSize.count(subspec)) {
        subspecBufferSize[subspec] = 0;
      }
      subspecBufferSize[subspec] += headerIn->payloadSize;
    }
  }

  /** loop over subspecs **/
  for (auto& subspecPartEntry : subspecPartMap) {

    auto subspec = subspecPartEntry.first;
    auto parts = subspecPartEntry.second;
    auto& firstPart = parts.at(0);

    /** use the first part to define output headers **/
    auto headerOut = *DataRefUtils::getHeader<o2::header::DataHeader*>(firstPart);
    auto dataProcessingHeaderOut = *DataRefUtils::getHeader<o2::framework::DataProcessingHeader*>(firstPart);
    headerOut.dataDescription = "CRAWDATA";
    headerOut.payloadSize = 0;

    /** initialise output message **/
    auto bufferSize = mOutputBufferSize > 0 ? mOutputBufferSize : subspecBufferSize[subspec];
    auto payloadMessage = device->NewMessage(bufferSize);
    auto bufferPointer = (char*)payloadMessage->GetData();

    /** loop over subspec parts **/
    for (const auto& ref : parts) {

      /** input **/
      auto headerIn = DataRefUtils::getHeader<o2::header::DataHeader*>(ref);
      auto dataProcessingHeaderIn = DataRefUtils::getHeader<o2::framework::DataProcessingHeader*>(ref);
      auto payloadIn = ref.payload;
      auto payloadInSize = headerIn->payloadSize;

      /** prepare compressor **/
      mCompressor.setDecoderBuffer(payloadIn);
      mCompressor.setDecoderBufferSize(payloadInSize);
      mCompressor.setEncoderBuffer(bufferPointer);
      mCompressor.setEncoderBufferSize(bufferSize);

      /** run **/
      mCompressor.run();
      auto payloadOutSize = mCompressor.getEncoderByteCounter();
      bufferPointer += payloadOutSize;
      bufferSize -= payloadOutSize;
      headerOut.payloadSize += payloadOutSize;
    }

    /** finalise output message **/
    payloadMessage->SetUsedSize(headerOut.payloadSize);
    o2::header::Stack headerStack{headerOut, dataProcessingHeaderOut};
    auto headerMessage = device->NewMessage(headerStack.size());
    std::memcpy(headerMessage->GetData(), headerStack.data(), headerStack.size());

    /** add parts **/
    partsOut.AddPart(std::move(headerMessage));
    partsOut.AddPart(std::move(payloadMessage));
  }

  /** send message **/
  device->Send(partsOut, fairMQChannel);
}

template class CompressorTask<o2::header::RAWDataHeaderV4, true>;
template class CompressorTask<o2::header::RAWDataHeaderV4, false>;
template class CompressorTask<o2::header::RAWDataHeaderV6, true>;
template class CompressorTask<o2::header::RAWDataHeaderV6, false>;

} // namespace tof
} // namespace o2
