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

/// @file   CruCompressorTask.cxx
/// @author Sean Murray
/// @brief  TRD cru output to tracklet task

#include "Framework/ControlService.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/RawDeviceService.h"
#include "Framework/DeviceSpec.h"
#include "Framework/DataSpecUtils.h"
#include "Headers/RDHAny.h"

#include "DataFormatsTRD/Constants.h"
#include "DataFormatsTRD/CompressedHeader.h"

#include "TRDReconstruction/CruCompressorTask.h"
#include "TRDReconstruction/CruRawReader.h"

#include <fairmq/Device.h>
#include <fairmq/Parts.h>
#include <iostream>

using namespace o2::framework;

namespace o2
{
namespace trd
{

void CruCompressorTask::init(InitContext& ic)
{
  LOG(info) << "FLP Compressore Task init";

  auto finishFunction = [this]() {
    mReader.checkSummary();
  };

  ic.services().get<CallbackService>().set(CallbackService::Id::Stop, finishFunction);
}

uint64_t CruCompressorTask::buildEventOutput()
{
  //mReader holds the vectors of tracklets and digits.
  // tracklets are 64 bit
  // digits are DigitMCMHeader and DigitMCMData

  uint64_t currentpos = 0;
  uint64_t trailer = 0xeeeeeeeeeeeeeeeeLL;
  //first we write a start rdh block
  CompressedRawHeader* trackletheader = (CompressedRawHeader*)&mOutBuffer[0];
  trackletheader->format = 1;
  trackletheader->eventtime = 99;
  currentpos = 1;
  //write the
  std::vector<o2::trd::TriggerRecord> ir;
  std::vector<o2::trd::Tracklet64> tracklets;
  std::vector<o2::trd::Digit> digits;
  mReader.getParsedObjects(tracklets, digits, ir);
  trackletheader->bc = ir[0].getBCData().bc;
  trackletheader->orbit = ir[0].getBCData().orbit;
  trackletheader->padding = 0xeeee;
  trackletheader->size = mReader.sumTrackletsFound() * 8; // to get to bytes. TODO compare to getTrackletsFound
  for (auto tracklet : mReader.getTracklets(ir[0].getBCData())) {
    //convert tracklet to 64 bit and add to blob
    mOutBuffer[currentpos++] = tracklet.getTrackletWord();
  }
  CompressedRawTrackletDigitSeperator* tracklettrailer = (CompressedRawTrackletDigitSeperator*)&mOutBuffer[currentpos];
  tracklettrailer->word = trailer;
  currentpos++;
  CompressedRawHeader* digitheader = (CompressedRawHeader*)&mOutBuffer[currentpos];
  currentpos++;
  digitheader->format = 2;
  digitheader->eventtime = 99;

  for (auto digit : mReader.getDigits(ir[0].getBCData())) {
    uint64_t* word;
    word = &mOutBuffer[currentpos];
    DigitMCMHeader mcmheader;
    mcmheader.eventcount = 1;
    mcmheader.mcm = digit.getMCM();
    mcmheader.rob = digit.getROB();
    mcmheader.yearflag = 1;
    mcmheader.eventcount = 1;
    mcmheader.res = 0xc; // formst is reservedto be 1100 binary
    //unpack the digit.
    mcmheader.word = (*word) >> 32;
    currentpos++;
  }
  CompressedRawTrackletDigitSeperator* digittrailer = (CompressedRawTrackletDigitSeperator*)&mOutBuffer[currentpos];
  digittrailer->word = trailer;
  currentpos++;
  //as far as I can tell this is almost always going to be blank.
  CompressedRawHeader* configheader = (CompressedRawHeader*)&mOutBuffer[currentpos];
  currentpos++;
  configheader->size = 2;
  configheader->format = 3;
  configheader->eventtime = 99;
  CompressedRawTrackletDigitSeperator* configtrailer = (CompressedRawTrackletDigitSeperator*)&mOutBuffer[currentpos];
  configtrailer->word = trailer;
  //finally we write a stop rdh block

  return currentpos;
}

void CruCompressorTask::run(ProcessingContext& pc)
{
  LOG(info) << "TRD Compression Task run method";

  auto device = pc.services().get<o2::framework::RawDeviceService>().device();
  auto outputRoutes = pc.services().get<o2::framework::RawDeviceService>().spec().outputs;
  auto fairMQChannel = outputRoutes.at(0).channel;
  int inputcount = 0;
  /* loop over inputs routes */
  for (auto iit = pc.inputs().begin(), iend = pc.inputs().end(); iit != iend; ++iit) {
    if (!iit.isValid()) {
      continue;
    }
    //LOG(info) << "iit.mInputs  " << iit.mInputs.
    /* prepare output parts */
    fair::mq::Parts parts;

    /* loop over input parts */
    for (auto const& ref : iit) {

      auto headerIn = DataRefUtils::getHeader<o2::header::DataHeader*>(ref);
      auto dataProcessingHeaderIn = DataRefUtils::getHeader<o2::framework::DataProcessingHeader*>(ref);
      auto payloadIn = ref.payload;
      auto payloadInSize = DataRefUtils::getPayloadSize(ref);
      std::cout << "payload In is : " << std::hex << payloadIn << std::endl;
      std::cout << "payload In is : " << std::dec << payloadIn << std::endl;
      std::cout << "payload In size is : " << std::dec << payloadInSize << std::endl;
      mReader.setDataBuffer(payloadIn);
      mReader.setDataBufferSize(payloadInSize);
      mReader.setVerbose(mVerbose);
      mReader.setDataVerbose(mDataVerbose);
      mReader.setHeaderVerbose(mHeaderVerbose);
      /* run */
      mReader.run();

      auto payloadOutSize = buildEventOutput();

      auto payloadOutSizeBytes = payloadOutSize * 8; // payloadoutsize in bytes.
      LOG(info) << "outgoing message has a data size of : " << payloadOutSize;
      if (payloadOutSizeBytes > 32 * 1024) {
        LOG(warn) << " buffer size for data is >32kB so will span rdh";
      }
      int numberofpiecestocutinto = payloadOutSizeBytes / (32 * 1024);
      int segmentsize = 32 * 1024 - 0x40; // 0x40 is the size of the rdh header in bytes.
      //the above will drop the decimal due to int.
      numberofpiecestocutinto++;
      for (int datasegment = 0; datasegment < numberofpiecestocutinto; ++datasegment) {
        auto payloadMessage = device->NewMessage(payloadOutSize);
        std::memcpy(payloadMessage->GetData(), (char*)mOutBuffer.data() + datasegment * segmentsize, payloadOutSize);
        /* output */
        auto headerOut = *headerIn;
        auto dataProcessingHeaderOut = *dataProcessingHeaderIn;
        headerOut.dataDescription = "CDATA";
        headerOut.payloadSize = payloadOutSizeBytes;
        // what to do about the packet count?
        //headerOut.packetCounter;
        o2::header::Stack headerStack{headerOut, dataProcessingHeaderOut};
        auto headerMessage = device->NewMessage(headerStack.size());
        std::memcpy(headerMessage->GetData(), headerStack.data(), headerStack.size());

        /* add parts */
        parts.AddPart(std::move(headerMessage));
        parts.AddPart(std::move(payloadMessage));
      }
    }

    /* send message */
    device->Send(parts, fairMQChannel);
  }
}

} // namespace trd
} // namespace o2
