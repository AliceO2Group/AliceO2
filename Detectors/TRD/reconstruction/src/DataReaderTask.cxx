// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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

void DataReaderTask::sendData(ProcessingContext& pc)
{

  if (mVerbose) {
    LOG(info) << "Sending data onwards with " << mDigits.size() << " Digits and " << mTracklets.size() << " Tracklets";
  }
  pc.outputs().snapshot(Output{o2::header::gDataOriginTRD, "DIGITS", 0, Lifetime::Timeframe}, mDigits);
  pc.outputs().snapshot(Output{o2::header::gDataOriginTRD, "TRACKLETS", 0, Lifetime::Timeframe}, mTracklets);
  pc.outputs().snapshot(Output{o2::header::gDataOriginTRD, "TRIGGERRECORD", 0, Lifetime::Timeframe}, mTriggers);
  //    pc.outputs().snapshot(Output{o2::header::gDataOriginTRD,"STATS",0,Lifetime::Timerframe},mStats);
}

void DataReaderTask::run(ProcessingContext& pc)
{
  LOG(info) << "TRD Translator Task run";
  auto dataReadStart = std::chrono::high_resolution_clock::now();
  /* set encoder output buffer */
  char bufferOut[o2::trd::constants::HBFBUFFERMAX];

  auto device = pc.services().get<o2::framework::RawDeviceService>().device();
  auto outputRoutes = pc.services().get<o2::framework::RawDeviceService>().spec().outputs;
  auto fairMQChannel = outputRoutes.at(0).channel;
  int inputcount = 0;
  /* loop over inputs routes */
  for (auto iit = pc.inputs().begin(), iend = pc.inputs().end(); iit != iend; ++iit) {
    if (!iit.isValid()) {
      continue;
    }
    /* loop over input parts */
    for (auto const& ref : iit) {

      const auto* headerIn = DataRefUtils::getHeader<o2::header::DataHeader*>(ref);
      auto payloadIn = ref.payload;
      auto payloadInSize = headerIn->payloadSize;
      if (!mCompressedData) {
        if (mVerbose) {
          LOG(info) << " parsing non compressed data in the data reader task";
        }

        int a = 1;
        int debugstopper = 1;
        //while(debugstopper==1){
        //  a=sin(rand());
        //}

        mReader.setDataBuffer(payloadIn);
        mReader.setDataBufferSize(payloadInSize);
        mReader.configure(mByteSwap, mVerbose, mHeaderVerbose, mDataVerbose);
        if (mVerbose) {
          LOG(info) << "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%";
          LOG(info) << "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%";
          LOG(info) << "%%%%%%%%%%%%%%%%%%%%%%%%%%% about to run %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%";
          LOG(info) << "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%";
          LOG(info) << "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%";
        }
        mReader.run();
        if (mVerbose) {
          LOG(info) << "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%";
          LOG(info) << "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%";
          LOG(info) << "%%%%%%%%%%%%%%%%%%%%%%%%%%% finished running %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%";
          LOG(info) << "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%";
          LOG(info) << "%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%";
        }
        mTracklets = mReader.getTracklets();
        mCompressedDigits = mReader.getCompressedDigits();
        if (mVerbose) {
          LOG(info) << "from parsing received: " << mTracklets.size() << " tracklets and " << mCompressedDigits.size() << " compressed digits";
        }
        mTriggers = mReader.getIR();
        //get the payload of trigger and digits out.
      } else { // we have compressed data coming in.
        mCompressedReader.setDataBuffer(payloadIn);
        mCompressedReader.setDataBufferSize(payloadInSize);
        mCompressedReader.configure(mByteSwap, mVerbose, mHeaderVerbose, mDataVerbose);
        mCompressedReader.run();
        mTracklets = mCompressedReader.getTracklets();
        mDigits = mCompressedReader.getDigits();
        mTriggers = mCompressedReader.getIR();
        //get the payload of trigger and digits out.
      }
      /* output */
      sendData(pc); //TODO do we ever have to not post the data. i.e. can we get here mid event? I dont think so.
    }
  }

  auto dataReadTime = std::chrono::high_resolution_clock::now() - dataReadStart;
  LOG(info) << "Processing time for Data reading  " << std::chrono::duration_cast<std::chrono::milliseconds>(dataReadTime).count() << "ms";
}

} // namespace o2::trd
