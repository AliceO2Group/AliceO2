// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   EpnDataReaderTask.cxx
/// @author Sean Murray
/// @brief  TRD cru output to tracklet task

#include "TRDReconstruction/EpnDataReaderTask.h"
#include "TRDReconstruction/CruRawReader.h"
#include "Framework/ControlService.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/RawDeviceService.h"
#include "Framework/DeviceSpec.h"
#include "Framework/DataSpecUtils.h"
#include "DataFormatsTRD/Constants.h"
#include <fairmq/FairMQDevice.h>

using namespace o2::framework;

namespace o2
{
namespace trd
{

void EpnDataReaderTask::init(InitContext& ic)
{
  LOG(INFO) << "EpnDataRead Task init";

  auto finishFunction = [this]() {
    mReader.checkSummary();
  };

  ic.services().get<CallbackService>().set(CallbackService::Id::Stop, finishFunction);
}

void EpnDataReaderTask::run(ProcessingContext& pc)
{
  LOG(info) << "TRD Translator Task run";

  /* set encoder output buffer */
  char bufferOut[o2::trd::constants::CRUBUFFERMAX];

  auto device = pc.services().get<o2::framework::RawDeviceService>().device();
  auto outputRoutes = pc.services().get<o2::framework::RawDeviceService>().spec().outputs;
  auto fairMQChannel = outputRoutes.at(0).channel;
  int inputcount = 0;
  /* loop over inputs routes */
  for (auto iit = pc.inputs().begin(), iend = pc.inputs().end(); iit != iend; ++iit) {
    if (!iit.isValid())
      continue;
    //LOG(info) << "iit.mInputs  " << iit.mInputs.
    /* prepare output parts */
    FairMQParts parts;

    /* loop over input parts */
    for (auto const& ref : iit) {

      auto headerIn = DataRefUtils::getHeader<o2::header::DataHeader*>(ref);
      auto dataProcessingHeaderIn = DataRefUtils::getHeader<o2::framework::DataProcessingHeader*>(ref);
      auto payloadIn = ref.payload;
      auto payloadInSize = headerIn->payloadSize;
      mReader.setDataBuffer(payloadIn);
      mReader.setDataBufferSize(payloadInSize);
      /* run */
      mReader.run();
      auto payloadOutSize = 1; //mReader.getEncoderByteCounter();
      auto payloadMessage = device->NewMessage(payloadOutSize);
      std::memcpy(payloadMessage->GetData(), bufferOut, payloadOutSize);

      /* output */
      auto headerOut = *headerIn;
      auto dataProcessingHeaderOut = *dataProcessingHeaderIn;
      headerOut.dataDescription = "TRDRAW";
      headerOut.payloadSize = payloadOutSize;
      o2::header::Stack headerStack{headerOut, dataProcessingHeaderOut};
      auto headerMessage = device->NewMessage(headerStack.size());
      std::memcpy(headerMessage->GetData(), headerStack.data(), headerStack.size());

      /* add parts */
      parts.AddPart(std::move(headerMessage));
      parts.AddPart(std::move(payloadMessage));
    }

    /* send message */
    device->Send(parts, fairMQChannel);
  }
}

} // namespace trd
} // namespace o2
