/**
 * FLPSender.cxx
 *
 * @since 2013-04-23
 * @author D. Klein, A. Rybalchenko, M. Al-Turany, C. Kouzinopoulos
 */

#include "FLPSender.h"
#include <assert.h>                                         // for assert
#include <stdint.h>
#include "FairMQMessage.h"
#include "FairMQTransportFactory.h"
#include "boost/date_time/posix_time/posix_time_types.hpp"
#include "boost/preprocessor/seq/enum.hpp"
#include "boost/preprocessor/seq/size.hpp"
#include "logger/logger.h"                                  // for LOG
class FairMQPoller;

using namespace std;
using boost::posix_time::ptime;

using namespace AliceO2::Devices;

struct f2eHeader {
  uint64_t timeFrameId;
  int      flpIndex;
};

FLPSender::FLPSender()
  : fHeartbeatTimeoutInMs(20000)
  , fOutputHeartbeat()
  , fIndex(0)
  , fSendOffset(0)
  , fSendDelay(8)
  , fHeaderBuffer()
  , fDataBuffer()
  , fEventSize(10000)
  , fTestMode(0)
{
}

FLPSender::~FLPSender()
{
}

void FLPSender::Init()
{
  ptime nullTime;

  for (int i = 0; i < fChannels["data-out"].size(); ++i) {
    fOutputHeartbeat.push_back(nullTime);
  }
}

bool FLPSender::updateIPHeartbeat(string reply)
{
  for (int i = 0; i < fChannels["data-out"].size(); ++i) {
    if (fChannels["data-out"].at(i).GetAddress() == reply) {
      ptime currentTime = boost::posix_time::microsec_clock::local_time();
      ptime storedHeartbeat = GetProperty(OutputHeartbeat, storedHeartbeat, i);

      if (to_simple_string(storedHeartbeat) != "not-a-date-time") {
        // LOG(INFO) << "EPN " << i << " (" << reply << ")" << " last seen "
        //           << (currentTime - storedHeartbeat).total_milliseconds() << " ms ago.";
      }
      else {
        LOG(INFO) << "IP has no heartbeat associated. Adding heartbeat: " << currentTime;
      }

      SetProperty(OutputHeartbeat, currentTime, i);

      return true;
    }
  }
  LOG(ERROR) << "IP " << reply << " unknown, not provided at execution time";

  return false;
}

void FLPSender::Run()
{
  FairMQPoller* poller = fTransportFactory->CreatePoller(fChannels["data-in"]);

  // base buffer, to be copied from for every timeframe body
  void* buffer = operator new[](fEventSize);
  FairMQMessage* baseMsg = fTransportFactory->CreateMessage(buffer, fEventSize);

  ptime currentTime;
  ptime storedHeartbeat;

  uint64_t timeFrameId = 0;

  while (GetCurrentState() == RUNNING) {
    poller->Poll(2);

    // input 0 - commands
    if (poller->CheckInput(0)) {
      FairMQMessage* commandMsg = fTransportFactory->CreateMessage();

      if (fChannels["data-in"].at(0).Receive(commandMsg) > 0) {
        //... handle command ...
      }

      delete commandMsg;
    }

    // input 1 - heartbeats
    if (poller->CheckInput(1)) {
      FairMQMessage* heartbeatMsg = fTransportFactory->CreateMessage();

      if (fChannels["data-in"].at(1).Receive(heartbeatMsg) > 0) {
        string reply = string(static_cast<char*>(heartbeatMsg->GetData()), heartbeatMsg->GetSize());
        updateIPHeartbeat(reply);
      }

      delete heartbeatMsg;
    }

    // input 2 - data (in test-mode: signal with a timeframe ID)
    if (poller->CheckInput(2)) {

      // initialize f2e header
      f2eHeader* h = new f2eHeader;

      if (fTestMode > 0) {
        // test-mode: receive and store id part in the buffer.
        FairMQMessage* idPart = fTransportFactory->CreateMessage();
        fChannels["data-in"].at(2).Receive(idPart);

        h->timeFrameId = *(reinterpret_cast<uint64_t*>(idPart->GetData()));
        h->flpIndex = fIndex;

        delete idPart;
      } else {
        // regular mode: use the id generated locally
        h->timeFrameId = timeFrameId;
        // h->flpIndex = stoi(fId);
        h->flpIndex = fIndex;

        if (++timeFrameId == UINT64_MAX - 1) {
          timeFrameId = 0;
        }
      }

      FairMQMessage* headerPart = fTransportFactory->CreateMessage(h, sizeof(f2eHeader));

      fHeaderBuffer.push(headerPart);

      // save the arrival time of the message.
      fArrivalTime.push(boost::posix_time::microsec_clock::local_time());

      if (fTestMode > 0) {
        // test-mode: initialize and store data part in the buffer.
        FairMQMessage* dataPart = fTransportFactory->CreateMessage();
        dataPart->Copy(baseMsg);
        fDataBuffer.push(dataPart);
      } else {
        // regular mode: receive data part from input
        FairMQMessage* dataPart = fTransportFactory->CreateMessage();
        fChannels["data-in"].at(2).Receive(dataPart);
        fDataBuffer.push(dataPart);
      }
    }

    // LOG(INFO) << fDataBuffer.size();

    // if offset is 0 - send data out without staggering.
    if (fSendOffset == 0 && fDataBuffer.size() > 0) {
      sendFrontData();
    } else if (fDataBuffer.size() > 0) {
      size_t dataSize = fDataBuffer.front()->GetSize();
      ptime now = boost::posix_time::microsec_clock::local_time();
      if ((now - fArrivalTime.front()).total_milliseconds() >= (fSendDelay * fSendOffset)) {
        sendFrontData();
      } else {
        // LOG(INFO) << "buffering...";
      }
    }
  }

  delete baseMsg;
}

inline void FLPSender::sendFrontData()
{
  f2eHeader h = *(reinterpret_cast<f2eHeader*>(fHeaderBuffer.front()->GetData()));
  uint64_t currentTimeframeId = h.timeFrameId;

  int SNDMORE = fChannels["data-in"].at(0).fSocket->SNDMORE;
  int NOBLOCK = fChannels["data-in"].at(0).fSocket->NOBLOCK;

  // for which EPN is the message?
  int direction = currentTimeframeId % fChannels["data-out"].size();
  // LOG(INFO) << "Sending event " << currentTimeframeId << " to EPN#" << direction << "...";

  ptime currentTime = boost::posix_time::microsec_clock::local_time();
  ptime storedHeartbeat = GetProperty(OutputHeartbeat, storedHeartbeat, direction);

  // if the heartbeat from the corresponding EPN is within timeout period, send the data.
  if (to_simple_string(storedHeartbeat) != "not-a-date-time" ||
      (currentTime - storedHeartbeat).total_milliseconds() < fHeartbeatTimeoutInMs) {
    if(fChannels["data-out"].at(direction).Send(fHeaderBuffer.front(), SNDMORE|NOBLOCK) == 0) {
      LOG(ERROR) << "Could not queue ID part of event #" << currentTimeframeId << " without blocking";
    }
    if (fChannels["data-out"].at(direction).Send(fDataBuffer.front(), NOBLOCK) == 0) {
      LOG(ERROR) << "Could not send message with event #" << currentTimeframeId << " without blocking";
    }
    fHeaderBuffer.pop();
    fArrivalTime.pop();
    fDataBuffer.pop();
  } else { // if the heartbeat is too old, discard the data.
    LOG(WARN) << "Heartbeat too old for EPN#" << direction << ", discarding message.";
    fHeaderBuffer.pop();
    fArrivalTime.pop();
    fDataBuffer.pop();
  }
}

void FLPSender::SetProperty(const int key, const string& value)
{
  switch (key) {
    default:
      FairMQDevice::SetProperty(key, value);
      break;
  }
}

string FLPSender::GetProperty(const int key, const string& default_/*= ""*/)
{
  switch (key) {
    default:
      return FairMQDevice::GetProperty(key, default_);
  }
}

void FLPSender::SetProperty(const int key, const int value)
{
  switch (key) {
    case HeartbeatTimeoutInMs:
      fHeartbeatTimeoutInMs = value;
      break;
    case Index:
      fIndex = value;
      break;
    case TestMode:
      fTestMode = value;
      break;
    case SendOffset:
      fSendOffset = value;
      break;
    case SendDelay:
      fSendDelay = value;
      break;
    case EventSize:
      fEventSize = value;
      break;
    default:
      FairMQDevice::SetProperty(key, value);
      break;
  }
}

int FLPSender::GetProperty(const int key, const int default_/*= 0*/)
{
  switch (key) {
    case HeartbeatTimeoutInMs:
      return fHeartbeatTimeoutInMs;
    case Index:
      return fIndex;
    case TestMode:
      return fTestMode;
    case SendOffset:
      return fSendOffset;
    case SendDelay:
      return fSendDelay;
    case EventSize:
      return fEventSize;
    default:
      return FairMQDevice::GetProperty(key, default_);
  }
}

// Method for setting properties represented as a heartbeat.
void FLPSender::SetProperty(const int key, const ptime value, const int slot /*= 0*/)
{
  switch (key) {
    case OutputHeartbeat:
      fOutputHeartbeat.erase(fOutputHeartbeat.begin() + slot);
      fOutputHeartbeat.insert(fOutputHeartbeat.begin() + slot, value);
      break;
  }
}

// Method for getting properties represented as a heartbeat.
ptime FLPSender::GetProperty(const int key, const ptime default_, const int slot /*= 0*/)
{
  switch (key) {
    case OutputHeartbeat:
      return fOutputHeartbeat.at(slot);
  }
  assert(false);
}
