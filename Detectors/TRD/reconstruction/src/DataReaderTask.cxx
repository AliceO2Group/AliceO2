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

/// @file   DataReaderTask.cxx
/// @author Sean Murray
/// @brief  TRD cru output to tracklet task

#include "TRDReconstruction/DataReaderTask.h"
#include "TRDReconstruction/CruRawReader.h"

#include "Framework/ControlService.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/RawDeviceService.h"
#include "Framework/DeviceSpec.h"
#include "Framework/DataSpecUtils.h"
#include "Framework/InputRecordWalker.h"

#include "DataFormatsTRD/Constants.h"

#include <fairmq/FairMQDevice.h>

//using namespace o2::framework;

namespace o2::trd
{

void DataReaderTask::init(InitContext& ic)
{
  LOG(INFO) << "o2::trd::DataReadTask init";

  auto finishFunction = [this]() {
    mReader.checkSummary();
  };

  ic.services().get<CallbackService>().set(CallbackService::Id::Stop, finishFunction);
  mDataDesc = "RAWDATA";
}

void DataReaderTask::sendData(ProcessingContext& pc, bool blankframe)
{
  if (!blankframe) {
    mReader.buildDPLOutputs(pc, mDataVerbose);
  } else {
    //ensure the objects we are sending back are indeed blank.
    //TODO maybe put this in buildDPLOutputs so sending all done in 1 place, not now though.
    std::vector<Tracklet64> tracklets;
    std::vector<Digit> digits;
    std::vector<o2::trd::TriggerRecord> triggers;
    LOG(info) << "Sending data onwards with " << digits.size() << " Digits and " << tracklets.size() << " Tracklets and " << triggers.size() << " Triggers and blankframe:" << blankframe;
    pc.outputs().snapshot(Output{o2::header::gDataOriginTRD, "DIGITS", 0, Lifetime::Timeframe}, digits);
    pc.outputs().snapshot(Output{o2::header::gDataOriginTRD, "TRACKLETS", 0, Lifetime::Timeframe}, tracklets);
    pc.outputs().snapshot(Output{o2::header::gDataOriginTRD, "TRKTRGRD", 0, Lifetime::Timeframe}, triggers);
  }
}

bool DataReaderTask::isTimeFrameEmpty(ProcessingContext& pc)
{
  constexpr auto origin = header::gDataOriginTRD;
  o2::framework::InputSpec dummy{"dummy",
                                 framework::ConcreteDataMatcher{origin,
                                                                header::gDataDescriptionRawData,
                                                                0xDEADBEEF}};
  // if we see requested data type input with 0xDEADBEEF subspec and 0 payload.
  // frame detected we have no data and send this instead
  // send empty output so as to not block workflow
  for (const auto& ref : o2::framework::InputRecordWalker(pc.inputs(), {dummy})) {
    const auto dh = o2::framework::DataRefUtils::getHeader<o2::header::DataHeader*>(ref);
    if (dh->payloadSize == 0) {
      LOGP(INFO, "Found blank input input [{}/{}/{:#x}] TF#{} 1st_orbit:{} Payload {} : ",
           dh->dataOrigin.str, dh->dataDescription.str, dh->subSpecification, dh->tfCounter, dh->firstTForbit, dh->payloadSize);
      return true;
    }
  }
  return false;
}

void DataReaderTask::run(ProcessingContext& pc)
{
  LOG(info) << "TRD Translator Task run";
  auto dataReadStart = std::chrono::high_resolution_clock::now();

  if (isTimeFrameEmpty(pc)) {
    sendData(pc, true); //send the empty tf data.
    return;
  }
  uint64_t total1 = 0, total2 = 0;
  /* set encoder output buffer */
  char bufferOut[o2::trd::constants::HBFBUFFERMAX];
  int loopcounter = 0;
  auto device = pc.services().get<o2::framework::RawDeviceService>().device();
  auto outputRoutes = pc.services().get<o2::framework::RawDeviceService>().spec().outputs;
  auto fairMQChannel = outputRoutes.at(0).channel;
  /* loop over inputs routes */
  for (auto iit = pc.inputs().begin(), iend = pc.inputs().end(); iit != iend; ++iit) {
    if (!iit.isValid()) {
      continue;
    }
    /* loop over input parts */
    int inputpartscount = 0;
    int emptyframe = 0;
    for (auto const& ref : iit) {
      auto inputprocessingstart = std::chrono::high_resolution_clock::now(); // measure total processing time
      if (mVerbose) {
        const auto dh = DataRefUtils::getHeader<o2::header::DataHeader*>(ref);
        LOGP(info, "Found input [{}/{}/{:#x}] TF#{} 1st_orbit:{} Payload {} : ",
             dh->dataOrigin.str, dh->dataDescription.str, dh->subSpecification, dh->tfCounter, dh->firstTForbit, dh->payloadSize);
      }
      const auto* headerIn = DataRefUtils::getHeader<o2::header::DataHeader*>(ref);
      auto payloadIn = ref.payload;
      auto payloadInSize = headerIn->payloadSize;
      //    const auto dh = DataRefUtils::getHeader<o2::header::DataHeader*>(ref);
      if (std::string(headerIn->dataDescription.str) != std::string("DISTSUBTIMEFRAMEFLP")) {
        if (!mCompressedData) { //we have raw data coming in from flp
          if (mVerbose) {
            LOG(info) << " parsing non compressed data in the data reader task with a payload of " << payloadInSize << " payload size";
          }
          //          LOG(info) << "start of data is at ref.payload=0x"<< std::hex << " headerIn->payloadSize:0x" << headerIn->payloadSize <<" headerIn->headerSize:0x" <<headerIn->headerSize;
          total1 += headerIn->payloadSize;
          total2 += headerIn->headerSize;
          //          LOG(info) << "start of data is at ref.payload=0x"<< std::hex << " total1:0x" << total1 <<" total2:0x" <<total2;
          mReader.setDataBuffer(payloadIn);
          mReader.setDataBufferSize(payloadInSize);
          mReader.configure(mByteSwap, mFixDigitEndCorruption, mTrackletHCHeaderState, mVerbose, mHeaderVerbose, mDataVerbose);
          mReader.run();
          mWordsRead += mReader.getWordsRead();
          mWordsRejected += mReader.getWordsRejected();
          if (mVerbose) {
            LOG(info) << "relevant vectors to read : " << mReader.sumTrackletsFound() << " tracklets and " << mReader.sumDigitsFound() << " compressed digits";
          }
        } else { // we have compressed data coming in from flp.
          mCompressedReader.setDataBuffer(payloadIn);
          mCompressedReader.setDataBufferSize(payloadInSize);
          mCompressedReader.configure(mByteSwap, mVerbose, mHeaderVerbose, mDataVerbose);
          mCompressedReader.run();
        }
      } // ignore the input of DISTSUBTIMEFRAMEFLP
        //      auto inputprocessingtime = std::chrono::high_resolution_clock::now() - inputprocessingstart;
        //     LOGP(info, "Input [{}/{}/{:#x}] TF#{} 1st_orbit:{} Payload {} : processed in {} us",
        //           dh->dataOrigin.str, dh->dataDescription.str, dh->subSpecification, dh->tfCounter, dh->firstTForbit, dh->payloadSize,std::chrono::duration_cast<std::chrono::microseconds>(inputprocessingtime).count());
    }
    /* output */
    sendData(pc, false);
  }

  auto dataReadTime = std::chrono::high_resolution_clock::now() - dataReadStart;
  LOG(info) << "Processing time for Data reading  " << std::chrono::duration_cast<std::chrono::microseconds>(dataReadTime).count() << "us";
  if (!mCompressedData) {
    LOG(info) << "Digits found : " << mReader.getDigitsFound();
    LOG(info) << "Tracklets found : " << mReader.getTrackletsFound();
    LOG(info) << "DataRead in :" << mWordsRead * 4 << " bytes";
    LOG(info) << "DataRejected in :" << mWordsRejected * 4 << " bytes";
    LOG(info) << "DataRetention :bad/good" << (double)mWordsRejected / (double)mWordsRead << "";
    LOG(info) << "Total % good data bad/(good+bad)" << (double)mWordsRejected / ((double)mWordsRead + (double)mWordsRejected) * 100.0 << " %";
  }
}

} // namespace o2::trd
