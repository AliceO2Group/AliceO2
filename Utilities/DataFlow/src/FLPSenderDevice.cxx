#include <cstdint>
#include <cassert>

#include "FairMQLogger.h"
#include "FairMQMessage.h"
#include "FairMQTransportFactory.h"
#include "FairMQProgOptions.h"

#include "Headers/DataHeader.h"
#include "Headers/SubframeMetadata.h"
#include "DataFlow/FLPSenderDevice.h"

using namespace std;
using namespace std::chrono;
using namespace AliceO2::Devices;
using SubframeMetadata = AliceO2::DataFlow::SubframeMetadata;

void FLPSenderDevice::InitTask()
{
  fIndex = GetConfig()->GetValue<int>("flp-index");
  fEventSize = GetConfig()->GetValue<int>("event-size");
  fNumEPNs = GetConfig()->GetValue<int>("num-epns");
  fTestMode = GetConfig()->GetValue<int>("test-mode");
  fSendOffset = GetConfig()->GetValue<int>("send-offset");
  fSendDelay = GetConfig()->GetValue<int>("send-delay");
  fInChannelName = GetConfig()->GetValue<string>("in-chan-name");
  fOutChannelName = GetConfig()->GetValue<string>("out-chan-name");
}


void FLPSenderDevice::Run()
{
  // base buffer, to be copied from for every timeframe body (zero-copy)
  FairMQMessagePtr baseMsg(NewMessage(fEventSize));

  // store the channel reference to avoid traversing the map on every loop iteration
  //FairMQChannel& dataInChannel = fChannels.at(fInChannelName).at(0);

  while (CheckCurrentState(RUNNING)) {
    // - Get the SubtimeframeMetadata
    // - Add the current FLP id to the SubtimeframeMetadata
    // - Forward to the EPN the whole subtimeframe
    FairMQParts subtimeframeParts;
    if (Receive(subtimeframeParts, fInChannelName, 0, 100) <= 0)
      continue;

    assert(subtimeframeParts.Size() != 0);
    assert(subtimeframeParts.Size() >= 2);
    Header::DataHeader* dh = reinterpret_cast<Header::DataHeader*>(subtimeframeParts.At(0)->GetData());
    assert(strncmp(dh->dataDescription.str, "SUBTIMEFRAMEMD", 16) == 0);

    SubframeMetadata* sfm = reinterpret_cast<SubframeMetadata*>(subtimeframeParts.At(1)->GetData());
    sfm->flpIndex = fIndex;

    fArrivalTime.push(steady_clock::now());
    fSTFBuffer.push(move(subtimeframeParts));

    // if offset is 0 - send data out without staggering.
    assert(fSTFBuffer.size() > 0);

    if (fSendOffset == 0 && fSTFBuffer.size() > 0) {
      sendFrontData();
    } else if (fSTFBuffer.size() > 0) {
      if (duration_cast<milliseconds>(steady_clock::now() - fArrivalTime.front()).count() >= (fSendDelay * fSendOffset)) {
        sendFrontData();
      } else {
        // LOG(INFO) << "buffering...";
      }
    }
  }
}

inline void FLPSenderDevice::sendFrontData()
{
  SubframeMetadata *sfm = static_cast<SubframeMetadata*>(fSTFBuffer.front().At(1)->GetData());
  uint16_t currentTimeframeId = AliceO2::DataFlow::timeframeIdFromTimestamp(sfm->startTime, sfm->duration);
  if (mLastTimeframeId != -1) {
    if (currentTimeframeId == mLastTimeframeId) {
      LOG(ERROR) << "Sent same consecutive timeframe ids\n";
    }
  }
  mLastTimeframeId = currentTimeframeId;

  // for which EPN is the message?
  int direction = currentTimeframeId % fNumEPNs;
  if (Send(fSTFBuffer.front(), fOutChannelName, direction, 0) < 0) {
    LOG(ERROR) << "Failed to queue sub-timeframe #" << currentTimeframeId << " to EPN[" << direction << "]";
  }
  fSTFBuffer.pop();
  fArrivalTime.pop();
}
