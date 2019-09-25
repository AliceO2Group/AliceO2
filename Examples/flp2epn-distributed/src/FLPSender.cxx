// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/**
 * FLPSender.cxx
 *
 * @since 2013-04-23
 * @author D. Klein, A. Rybalchenko, M. Al-Turany, C. Kouzinopoulos
 */

#include <cstdint> // UINT64_MAX
#include <cassert>

#include <FairMQLogger.h>
#include <FairMQMessage.h>
#include <options/FairMQProgOptions.h>

#include "FLP2EPNex_distributed/FLPSender.h"
#include "O2Device/Compatibility.h"

using namespace std;
using namespace std::chrono;
using namespace o2::devices;

struct f2eHeader {
  uint16_t timeFrameId;
  int flpIndex;
};

FLPSender::FLPSender()
  : mSTFBuffer(), mArrivalTime(), mNumEPNs(0), mIndex(0), mSendOffset(0), mSendDelay(8), mEventSize(10000), mTestMode(0), mTimeFrameId(0), mInChannelName(), mOutChannelName()
{
}

FLPSender::~FLPSender() = default;

void FLPSender::InitTask()
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

void FLPSender::Run()
{
  // base buffer, to be copied from for every timeframe body (zero-copy)
  FairMQMessagePtr baseMsg(NewMessage(mEventSize));

  // store the channel reference to avoid traversing the map on every loop iteration
  FairMQChannel& dataInChannel = fChannels.at(mInChannelName).at(0);

  while (compatibility::FairMQ13<FairMQDevice>::IsRunning(this)) {
    // initialize f2e header
    auto* header = new f2eHeader;
    if (mTestMode > 0) {
      // test-mode: receive and store id part in the buffer.
      FairMQMessagePtr id(NewMessage());
      if (dataInChannel.Receive(id) > 0) {
        header->timeFrameId = *(static_cast<uint16_t*>(id->GetData()));
        header->flpIndex = mIndex;
      } else {
        // if nothing was received, try again
        delete header;
        continue;
      }
    } else {
      // regular mode: use the id generated locally
      header->timeFrameId = mTimeFrameId;
      header->flpIndex = mIndex;

      if (++mTimeFrameId == UINT16_MAX - 1) {
        mTimeFrameId = 0;
      }
    }

    FairMQParts parts;

    parts.AddPart(NewMessage(
      header, sizeof(f2eHeader), [](void* data, void* hint) { delete static_cast<f2eHeader*>(hint); }, header));
    parts.AddPart(NewMessage());

    // save the arrival time of the message.
    mArrivalTime.push(steady_clock::now());

    if (mTestMode > 0) {
      // test-mode: initialize and store data part in the buffer.
      parts.At(1)->Copy(*baseMsg);
      mSTFBuffer.push(move(parts));
    } else {
      // regular mode: receive data part from input
      if (dataInChannel.Receive(parts.At(1)) >= 0) {
        mSTFBuffer.push(move(parts));
      } else {
        // if nothing was received, try again
        continue;
      }
    }

    // if offset is 0 - send data out without staggering.
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

inline void FLPSender::sendFrontData()
{
  f2eHeader header = *(static_cast<f2eHeader*>(mSTFBuffer.front().At(0)->GetData()));
  uint16_t currentTimeframeId = header.timeFrameId;

  // for which EPN is the message?
  int direction = currentTimeframeId % mNumEPNs;
  // LOG(INFO) << "Sending event " << currentTimeframeId << " to EPN#" << direction << "...";

  if (Send(mSTFBuffer.front(), mOutChannelName, direction, 0) < 0) {
    LOG(ERROR) << "Failed to queue sub-timeframe #" << currentTimeframeId << " to EPN[" << direction << "]";
  }
  mSTFBuffer.pop();
  mArrivalTime.pop();
}
