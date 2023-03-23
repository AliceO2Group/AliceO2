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

/// @file   CompressorTaskOld.cxx
/// @author Roberto Preghenella
/// @since  2019-12-18
/// @brief  TOF raw data compressor task

#include "TOFCompression/CompressorTaskOld.h"
#include "Framework/ControlService.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/RawDeviceService.h"
#include "Framework/DeviceSpec.h"
#include "Framework/DataSpecUtils.h"
#include "Framework/InputRecordWalker.h"
#include "CommonUtils/VerbosityConfig.h"

#include <fairmq/Device.h>
#include <fairmq/Parts.h>

using namespace o2::framework;

namespace o2
{
namespace tof
{

template <typename RDH, bool verbose, bool paranoid>
void CompressorTaskOld<RDH, verbose, paranoid>::init(InitContext& ic)
{
  LOG(info) << "Compressor init";

  auto decoderCONET = ic.options().get<bool>("tof-compressor-conet-mode");
  auto decoderVerbose = ic.options().get<bool>("tof-compressor-decoder-verbose");
  auto encoderVerbose = ic.options().get<bool>("tof-compressor-encoder-verbose");
  auto checkerVerbose = ic.options().get<bool>("tof-compressor-checker-verbose");
  mOutputBufferSize = ic.options().get<int>("tof-compressor-output-buffer-size");

  mCompressor.setDecoderCONET(decoderCONET);
  mCompressor.setDecoderVerbose(decoderVerbose);
  mCompressor.setEncoderVerbose(encoderVerbose);
  mCompressor.setCheckerVerbose(checkerVerbose);

  auto finishFunction = [this]() {
    mCompressor.checkSummary();
  };

  ic.services().get<CallbackService>().set<CallbackService::Id::Stop>(finishFunction);
}

template <typename RDH, bool verbose, bool paranoid>
void CompressorTaskOld<RDH, verbose, paranoid>::run(ProcessingContext& pc)
{
  LOG(debug) << "Compressor run";

  auto device = pc.services().get<o2::framework::RawDeviceService>().device();
  auto outputRoutes = pc.services().get<o2::framework::RawDeviceService>().spec().outputs;
  if (outputRoutes.size() != 1) {
    LOG(error) << "Compressor output routes size != 1";
    return;
  }
  auto fairMQChannel = outputRoutes.at(0).channel;
  fair::mq::Parts partsOut;

  /** to store data sorted by subspec id **/
  std::map<int, std::vector<o2::framework::DataRef>> subspecPartMap;
  std::map<int, int> subspecBufferSize;

  // if we see requested data type input with 0xDEADBEEF subspec and 0 payload this means that the "delayed message"
  // mechanism created it in absence of real data from upstream. Processor should send empty output to not block the workflow
  {
    auto& inputs = pc.inputs();
    static size_t contDeadBeef = 0; // number of times 0xDEADBEEF was seen continuously
    std::vector<InputSpec> dummy{InputSpec{"dummy", ConcreteDataMatcher{"TOF", "RAWDATA", 0xDEADBEEF}}};
    for (const auto& ref : InputRecordWalker(inputs, dummy)) {
      const auto* dh = o2::framework::DataRefUtils::getHeader<o2::header::DataHeader*>(ref);
      auto payloadSize = DataRefUtils::getPayloadSize(ref);
      if (payloadSize == 0) {
        auto maxWarn = o2::conf::VerbosityConfig::Instance().maxWarnDeadBeef;
        if (++contDeadBeef <= maxWarn) {
          LOGP(alarm, "Found input [{}/{}/{:#x}] TF#{} 1st_orbit:{} Payload {} : assuming no payload for all links in this TF{}",
               dh->dataOrigin.str, dh->dataDescription.str, dh->subSpecification, dh->tfCounter, dh->firstTForbit, payloadSize,
               contDeadBeef == maxWarn ? fmt::format(". {} such inputs in row received, stopping reporting", contDeadBeef) : "");
        }
        // send empty message with DEADBEEF subspec
        const auto* dph = o2::framework::DataRefUtils::getHeader<o2::framework::DataProcessingHeader*>(ref);
        o2::header::DataHeader emptyDH("CRAWDATA", "TOF", 0xdeadbeef, 0, 0, 1);
        emptyDH.runNumber = dh->runNumber;
        emptyDH.payloadSerializationMethod = o2::header::gSerializationMethodNone;
        emptyDH.firstTForbit = dh->firstTForbit;
        emptyDH.tfCounter = dh->tfCounter;

        o2::header::Stack emptyStack{emptyDH, o2::framework::DataProcessingHeader{dph->startTime, dph->duration, dph->creation}};

        auto headerMessage = device->NewMessage(emptyStack.size());
        auto payloadMessage = device->NewMessage(0);
        std::memcpy(headerMessage->GetData(), emptyStack.data(), emptyStack.size());
        partsOut.AddPart(std::move(headerMessage));
        partsOut.AddPart(std::move(payloadMessage));
        device->Send(partsOut, fairMQChannel);
        return;
      }
    }
    contDeadBeef = 0; // if good data, reset the counter
  }

  /** loop over inputs routes **/
  std::vector<InputSpec> sel{InputSpec{"filter", ConcreteDataTypeMatcher{"TOF", "RAWDATA"}}};
  for (const auto& ref : InputRecordWalker(pc.inputs(), sel)) {
    //  for (auto iit = pc.inputs().begin(), iend = pc.inputs().end(); iit != iend; ++iit) {
    //    if (!iit.isValid()) {
    //      continue;
    //    }

    /** loop over input parts **/
    //    for (auto const& ref : iit) {

    /** store parts in map **/
    auto headerIn = DataRefUtils::getHeader<o2::header::DataHeader*>(ref);
    auto payloadInSize = DataRefUtils::getPayloadSize(ref);
    auto subspec = headerIn->subSpecification;
    subspecPartMap[subspec].push_back(ref);

    /** increase subspec buffer size **/
    if (!subspecBufferSize.count(subspec)) {
      subspecBufferSize[subspec] = 0;
    }
    subspecBufferSize[subspec] += payloadInSize;
    //  }
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
    headerOut.splitPayloadParts = 1;

    /** initialise output message **/
    auto bufferSize = mOutputBufferSize >= 0 ? mOutputBufferSize + subspecBufferSize[subspec] : std::abs(mOutputBufferSize);
    auto bufferSizeDouble = bufferSize * 2;
    auto payloadMessage = device->NewMessage(bufferSizeDouble); // start with double size since later resized (but resize cannot resize more than original one)
    auto bufferPointer = (char*)payloadMessage->GetData();

    /** loop over subspec parts **/
    for (const auto& ref : parts) {

      /** input **/
      auto headerIn = DataRefUtils::getHeader<o2::header::DataHeader*>(ref);
      auto dataProcessingHeaderIn = DataRefUtils::getHeader<o2::framework::DataProcessingHeader*>(ref);
      auto payloadIn = ref.payload;
      auto payloadInSize = DataRefUtils::getPayloadSize(ref);

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
    if (headerOut.payloadSize > bufferSizeDouble) {
      headerOut.payloadSize = 0; // put payload to zero, otherwise it will trigger a crash
    }
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

template class CompressorTaskOld<o2::header::RAWDataHeader, false, false>;
template class CompressorTaskOld<o2::header::RAWDataHeader, false, true>;
template class CompressorTaskOld<o2::header::RAWDataHeader, true, false>;
template class CompressorTaskOld<o2::header::RAWDataHeader, true, true>;

} // namespace tof
} // namespace o2
