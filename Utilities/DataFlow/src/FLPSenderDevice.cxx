// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include <cstdint>
#include <cassert>

#include <FairMQLogger.h>
#include <FairMQMessage.h>
#include <options/FairMQProgOptions.h>

#include "Headers/DataHeader.h"
#include "Headers/SubframeMetadata.h"
#include "DataFlow/FLPSenderDevice.h"
#include "O2Device/Compatibility.h"

using namespace std;
using namespace std::chrono;
using namespace o2::devices;
using SubframeMetadata = o2::data_flow::SubframeMetadata;

void FLPSenderDevice::InitTask()
{
  mIndex = GetConfig()->GetValue<int>("flp-index");
  mEventSize = GetConfig()->GetValue<int>("event-size");
  mNumEPNs = GetConfig()->GetValue<int>("num-epns");
  mTestMode = GetConfig()->GetValue<int>("test-mode");
  mSendOffset = GetConfig()->GetValue<int>("send-offset");
  mSendDelay = GetConfig()->GetValue<int>("send-delay");
  mInChannelName = GetConfig()->GetValue<string>("in-chan-name");
  mOutChannelName = GetConfig()->GetValue<string>("out-chan-name");
}

void FLPSenderDevice::Run()
{
  // base buffer, to be copied from for every timeframe body (zero-copy)
  FairMQMessagePtr baseMsg(NewMessage(mEventSize));

  // store the channel reference to avoid traversing the map on every loop iteration
  //FairMQChannel& dataInChannel = fChannels.at(fInChannelName).at(0);

  while (compatibility::FairMQ13<FairMQDevice>::IsRunning(this)) {
    // - Get the SubtimeframeMetadata
    // - Add the current FLP id to the SubtimeframeMetadata
    // - Forward to the EPN the whole subtimeframe
    FairMQParts subtimeframeParts;
    if (Receive(subtimeframeParts, mInChannelName, 0, 100) <= 0) {
      continue;
    }

    assert(subtimeframeParts.Size() != 0);
    assert(subtimeframeParts.Size() >= 2);
    const auto* dh = o2::header::get<header::DataHeader*>(subtimeframeParts.At(0)->GetData());
    assert(strncmp(dh->dataDescription.str, "SUBTIMEFRAMEMD", 16) == 0);

    SubframeMetadata* sfm = reinterpret_cast<SubframeMetadata*>(subtimeframeParts.At(1)->GetData());
    sfm->flpIndex = mIndex;

    mArrivalTime.push(steady_clock::now());
    mSTFBuffer.push(move(subtimeframeParts));

    // if offset is 0 - send data out without staggering.
    assert(mSTFBuffer.size() > 0);

    if (mSendOffset == 0 && mSTFBuffer.size() > 0) {
      sendFrontData();
    } else if (mSTFBuffer.size() > 0) {
      if (duration_cast<milliseconds>(steady_clock::now() - mArrivalTime.front()).count() >= (mSendDelay * mSendOffset)) {
        sendFrontData();
      } else {
        // LOG(INFO) << "buffering...";
      }
    }
  }
}

inline void FLPSenderDevice::sendFrontData()
{
  SubframeMetadata* sfm = static_cast<SubframeMetadata*>(mSTFBuffer.front().At(1)->GetData());
  uint16_t currentTimeframeId = o2::data_flow::timeframeIdFromTimestamp(sfm->startTime, sfm->duration);
  if (mLastTimeframeId != -1) {
    if (currentTimeframeId == mLastTimeframeId) {
      LOG(ERROR) << "Sent same consecutive timeframe ids\n";
    }
  }
  mLastTimeframeId = currentTimeframeId;

  // for which EPN is the message?
  int direction = currentTimeframeId % mNumEPNs;
  if (Send(mSTFBuffer.front(), mOutChannelName, direction, 0) < 0) {
    LOG(ERROR) << "Failed to queue sub-timeframe #" << currentTimeframeId << " to EPN[" << direction << "]";
  }
  mSTFBuffer.pop();
  mArrivalTime.pop();
}
