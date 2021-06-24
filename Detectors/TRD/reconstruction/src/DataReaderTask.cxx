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
}

void DataReaderTask::sendData(ProcessingContext& pc, bool blankframe)
{
  // mReader.getParsedObjects(mTracklets,mDigits,mTriggers);
  if (!blankframe) {
    mReader.getParsedObjects(mTracklets, mDigits, mTriggers);
  }

  LOG(info) << "Sending data onwards with " << mDigits.size() << " Digits and " << mTracklets.size() << " Tracklets and " << mTriggers.size() << " Triggers and blankframe:" << blankframe;
  pc.outputs().snapshot(Output{o2::header::gDataOriginTRD, "DIGITS", 0, Lifetime::Timeframe}, mDigits);
  pc.outputs().snapshot(Output{o2::header::gDataOriginTRD, "TRACKLETS", 0, Lifetime::Timeframe}, mTracklets);
  pc.outputs().snapshot(Output{o2::header::gDataOriginTRD, "TRKTRGRD", 0, Lifetime::Timeframe}, mTriggers);
  //    pc.outputs().snapshot(Output{o2::header::gDataOriginTRD,"STATS",0,Lifetime::Timerframe},mStats);
}

void DataReaderTask::run(ProcessingContext& pc)
{
  LOG(info) << "TRD Translator Task run";
  auto dataReadStart = std::chrono::high_resolution_clock::now();
  /* set encoder output buffer */
  char bufferOut[o2::trd::constants::HBFBUFFERMAX];
  int loopcounter = 0;
  auto device = pc.services().get<o2::framework::RawDeviceService>().device();
  auto outputRoutes = pc.services().get<o2::framework::RawDeviceService>().spec().outputs;
  auto fairMQChannel = outputRoutes.at(0).channel;
  mDataSpec = o2::header::gDataDescriptionRawData;

  std::vector<InputSpec> dummy{InputSpec{"dummy", ConcreteDataMatcher{"TRD", mDataSpec, 0xDEADBEEF}}};
  // if we see requested data type input with 0xDEADBEEF subspec and 0 payload this means that the "delayed message"
  //   // mechanism created it in absence of real data from upstream. Processor should send empty output to not block the workflow

  for (const auto& ref : InputRecordWalker(pc.inputs(), dummy)) {
    const auto dh = o2::framework::DataRefUtils::getHeader<o2::header::DataHeader*>(ref);
    if (dh->payloadSize == 16 || dh->payloadSize == 0) {
      LOGP(WARNING, "Found input [{}/{}/{:#x}] TF#{} 1st_orbit:{} Payload {} : assuming no payload for all links in this TF",
           dh->dataOrigin.str, dh->dataDescription.str, dh->subSpecification, dh->tfCounter, dh->firstTForbit, dh->payloadSize);
      sendData(pc, true); //send the empty tf data.
      return;
    }
    LOG(info) << " matched DEADBEEF";
  }
  //TODO combine the previous and subsequent loops.
  /* loop over inputs routes */
  for (auto iit = pc.inputs().begin(), iend = pc.inputs().end(); iit != iend; ++iit) {
    if (!iit.isValid()) {
      continue;
    }
    /* loop over input parts */
    int inputpartscount = 0;
    for (auto const& ref : iit) {
      if (mVerbose) {
        const auto dh = DataRefUtils::getHeader<o2::header::DataHeader*>(ref);
        LOGP(info, "Found input [{}/{}/{:#x}] TF#{} 1st_orbit:{} Payload {} : assuming no payload for all links in this TF",
             dh->dataOrigin.str, dh->dataDescription.str, dh->subSpecification, dh->tfCounter, dh->firstTForbit, dh->payloadSize);
      }
      const auto* headerIn = DataRefUtils::getHeader<o2::header::DataHeader*>(ref);
      auto payloadIn = ref.payload;
      auto payloadInSize = headerIn->payloadSize;
      if (!mCompressedData) { //we have raw data coming in from flp
        if (mVerbose) {
          LOG(info) << " parsing non compressed data in the data reader task with a payload of " << payloadInSize << " payload size";
        }
        mReader.setDataBuffer(payloadIn);
        mReader.setDataBufferSize(payloadInSize);
        mReader.configure(mByteSwap, mVerbose, mHeaderVerbose, mDataVerbose);
        if (mVerbose) {
          LOG(info) << "%%% about to run " << loopcounter << " %%%";
        }
        mReader.run();
        if (mVerbose) {
          LOG(info) << "%%% finished running " << loopcounter << " %%%";
        }
        loopcounter++;
        // mTracklets.insert(std::end(mTracklets), std::begin(mReader.getTracklets()), std::end(mReader.getTracklets()));
        // mCompressedDigits.insert(std::end(mCompressedDigits), std::begin(mReader.getCompressedDigits()), std::end(mReader.getCompressedDigits()));
        //mReader.clearall();
        if (mVerbose) {
          LOG(info) << "from parsing received: " << mTracklets.size() << " tracklets and " << mDigits.size() << " compressed digits";
          LOG(info) << "relevant vectors to read : " << mReader.sumTrackletsFound() << " tracklets and " << mReader.sumDigitsFound() << " compressed digits";
        }
        //  mTriggers = mReader.getIR();
        //get the payload of trigger and digits out.
      } else { // we have compressed data coming in from flp.
        mCompressedReader.setDataBuffer(payloadIn);
        mCompressedReader.setDataBufferSize(payloadInSize);
        mCompressedReader.configure(mByteSwap, mVerbose, mHeaderVerbose, mDataVerbose);
        mCompressedReader.run();
        mTracklets = mCompressedReader.getTracklets();
        mDigits = mCompressedReader.getDigits();
        mTriggers = mCompressedReader.getIR();
        //get the payload of trigger and digits out.
      }
    }
    /* output */
    sendData(pc, false); //TODO do we ever have to not post the data. i.e. can we get here mid event? I dont think so.
  }

  auto dataReadTime = std::chrono::high_resolution_clock::now() - dataReadStart;
  LOG(info) << "Processing time for Data reading  " << std::chrono::duration_cast<std::chrono::milliseconds>(dataReadTime).count() << "ms";
  if (!mCompressedData) {
    LOG(info) << "Digits found : " << mReader.getDigitsFound();
    LOG(info) << "Digits returned : " << mDigits.size();

    LOG(info) << "Tracklets found : " << mReader.getTrackletsFound();
    LOG(info) << "Tracklets returned : " << mTracklets.size();
  }
}

} // namespace o2::trd
