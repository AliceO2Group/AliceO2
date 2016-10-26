/**
 * FLPSender.cxx
 *
 * @since 2013-04-23
 * @author D. Klein, A. Rybalchenko, M. Al-Turany, C. Kouzinopoulos
 */

#include <cstdint> // UINT64_MAX
#include <cassert>

#include "FairMQLogger.h"
#include "FairMQMessage.h"
#include "FairMQTransportFactory.h"
#include "FairMQProgOptions.h"

#include "FLP2EPNex_distributed/FLPSender.h"

using namespace std;
using namespace std::chrono;
using namespace AliceO2::Devices;

struct f2eHeader {
  uint16_t timeFrameId;
  int      flpIndex;
};

FLPSender::FLPSender()
  : fSTFBuffer()
  , fArrivalTime()
  , fNumEPNs(0)
  , fIndex(0)
  , fSendOffset(0)
  , fSendDelay(8)
  , fEventSize(10000)
  , fTestMode(0)
  , fTimeFrameId(0)
  , fInChannelName()
  , fOutChannelName()
{
}

FLPSender::~FLPSender()
{
}

void FLPSender::InitTask()
{
  fIndex = fConfig->GetValue<int>("flp-index");
  fEventSize = fConfig->GetValue<int>("event-size");
  fNumEPNs = fConfig->GetValue<int>("num-epns");
  fTestMode = fConfig->GetValue<int>("test-mode");
  fSendOffset = fConfig->GetValue<int>("send-offset");
  fSendDelay = fConfig->GetValue<int>("send-delay");
  fInChannelName = fConfig->GetValue<string>("in-chan-name");
  fOutChannelName = fConfig->GetValue<string>("out-chan-name");
}

void FLPSender::Run()
{
  // base buffer, to be copied from for every timeframe body (zero-copy)
  FairMQMessagePtr baseMsg(NewMessage(fEventSize));

  // store the channel reference to avoid traversing the map on every loop iteration
  FairMQChannel& dataInChannel = fChannels.at(fInChannelName).at(0);

  while (CheckCurrentState(RUNNING)) {
    // initialize f2e header
    f2eHeader* header = new f2eHeader;
    if (fTestMode > 0) {
      // test-mode: receive and store id part in the buffer.
      FairMQMessagePtr id(NewMessage());
      if (dataInChannel.Receive(id) > 0) {
        header->timeFrameId = *(static_cast<uint16_t*>(id->GetData()));
        header->flpIndex = fIndex;
      } else {
        // if nothing was received, try again
        delete header;
        continue;
      }
    } else {
      // regular mode: use the id generated locally
      header->timeFrameId = fTimeFrameId;
      header->flpIndex = fIndex;

      if (++fTimeFrameId == UINT16_MAX - 1) {
        fTimeFrameId = 0;
      }
    }

    FairMQParts parts;

    parts.AddPart(NewMessage(header, sizeof(f2eHeader), [](void* data, void* hint){ delete static_cast<f2eHeader*>(hint); }, header));
    parts.AddPart(NewMessage());

    // save the arrival time of the message.
    fArrivalTime.push(steady_clock::now());

    if (fTestMode > 0) {
      // test-mode: initialize and store data part in the buffer.
      parts.At(1)->Copy(baseMsg);
      fSTFBuffer.push(move(parts));
    } else {
      // regular mode: receive data part from input
      if (dataInChannel.Receive(parts.At(1)) >= 0) {
        fSTFBuffer.push(move(parts));
      } else {
        // if nothing was received, try again
        continue;
      }
    }

    // if offset is 0 - send data out without staggering.
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

inline void FLPSender::sendFrontData()
{
  f2eHeader header = *(static_cast<f2eHeader*>(fSTFBuffer.front().At(0)->GetData()));
  uint16_t currentTimeframeId = header.timeFrameId;

  // for which EPN is the message?
  int direction = currentTimeframeId % fNumEPNs;
  // LOG(INFO) << "Sending event " << currentTimeframeId << " to EPN#" << direction << "...";

  if (Send(fSTFBuffer.front(), fOutChannelName, direction, 0) < 0) {
    LOG(ERROR) << "Failed to queue sub-timeframe #" << currentTimeframeId << " to EPN[" << direction << "]";
  }
  fSTFBuffer.pop();
  fArrivalTime.pop();
}
