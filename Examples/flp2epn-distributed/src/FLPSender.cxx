/**
 * FLPSender.cxx
 *
 * @since 2013-04-23
 * @author D. Klein, A. Rybalchenko, M. Al-Turany, C. Kouzinopoulos
 */

#include <cstdint> // UINT64_MAX
#include <cassert>

#include <boost/thread.hpp>
#include <boost/bind.hpp>

#include "FairMQLogger.h"
#include "FairMQMessage.h"
#include "FairMQTransportFactory.h"

#include "FLP2EPNex_distributed/FLPSender.h"

using namespace std;
using boost::posix_time::ptime;

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
  , fHeartbeatTimeoutInMs(20000)
  , fHeartbeats()
  , fHeartbeatMutex()
{
}

FLPSender::~FLPSender()
{
}

void FLPSender::InitTask()
{
  ptime nullTime;

  fNumEPNs = fChannels.at("data-out").size();

  for (int i = 0; i < fNumEPNs; ++i) {
    fHeartbeats[fChannels.at("data-out").at(i).GetAddress()] = nullTime;
  }
}

void FLPSender::receiveHeartbeats()
{
  FairMQChannel& hbChannel = fChannels.at("heartbeat-in").at(0);

  while (CheckCurrentState(RUNNING)) {
    try {
      unique_ptr<FairMQMessage> heartbeat(NewMessage());

      if (hbChannel.Receive(heartbeat) > 0) {
        string address = string(static_cast<char*>(heartbeat->GetData()), heartbeat->GetSize());

        if (fHeartbeats.find(address) != fHeartbeats.end()) {
            ptime now = boost::posix_time::microsec_clock::local_time();
            ptime storedHeartbeat = fHeartbeats[address];

            if (to_simple_string(storedHeartbeat) != "not-a-date-time") {
              // LOG(INFO) << address << " last seen " << (now - storedHeartbeat).total_milliseconds() << " ms ago.";
            } else {
              LOG(INFO) << "Received first heartbeat from " << address << " (" << now << ")";
            }

            boost::unique_lock<boost::shared_mutex> uniqueLock(fHeartbeatMutex);
            fHeartbeats[address] = now;
        } else {
          LOG(ERROR) << "IP " << address << " unknown, not provided at execution time";
        }
      }
    } catch (boost::thread_interrupted&) {
      LOG(INFO) << "FLPSender::receiveHeartbeats() interrupted";
      break;
    }
  }
}

void FLPSender::Run()
{
  // boost::thread heartbeatReceiver(boost::bind(&FLPSender::receiveHeartbeats, this));

  // base buffer, to be copied from for every timeframe body (zero-copy)
  unique_ptr<FairMQMessage> baseMsg(NewMessage(fEventSize));

  uint16_t timeFrameId = 0;

  // store the channel reference to avoid traversing the map on every loop iteration
  FairMQChannel& dataInChannel = fChannels.at("data-in").at(0);

  while (CheckCurrentState(RUNNING)) {
    // initialize f2e header
    f2eHeader* header = new f2eHeader;
    if (fTestMode > 0) {
      // test-mode: receive and store id part in the buffer.
      unique_ptr<FairMQMessage> id(NewMessage());
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
      header->timeFrameId = timeFrameId;
      header->flpIndex = fIndex;

      if (++timeFrameId == UINT16_MAX - 1) {
        timeFrameId = 0;
      }
    }

    FairMQParts parts;

    parts.AddPart(NewMessage(header, sizeof(f2eHeader), [](void* data, void* hint){ delete static_cast<f2eHeader*>(hint); }, header));
    parts.AddPart(NewMessage());

    // save the arrival time of the message.
    fArrivalTime.push(boost::posix_time::microsec_clock::local_time());

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
      ptime now = boost::posix_time::microsec_clock::local_time();
      if ((now - fArrivalTime.front()).total_milliseconds() >= (fSendDelay * fSendOffset)) {
        sendFrontData();
      } else {
        // LOG(INFO) << "buffering...";
      }
    }
  }

  // heartbeatReceiver.interrupt();
  // heartbeatReceiver.join();
}

inline void FLPSender::sendFrontData()
{
  f2eHeader header = *(static_cast<f2eHeader*>(fSTFBuffer.front().At(0)->GetData()));
  uint16_t currentTimeframeId = header.timeFrameId;

  // for which EPN is the message?
  int direction = currentTimeframeId % fNumEPNs;
  // LOG(INFO) << "Sending event " << currentTimeframeId << " to EPN#" << direction << "...";

  // // get current time to compare with the latest heartbeat from destination EPN.
  // ptime currentTime = boost::posix_time::microsec_clock::local_time();
  // ptime storedHeartbeat;
  // {
  //   boost::shared_lock<boost::shared_mutex> lock(fHeartbeatMutex);
  //   storedHeartbeat = fHeartbeats[fChannels.at("data-out").at(direction).GetAddress()];
  // }

  // // if the heartbeat is too old, discard the data.
  // if (to_simple_string(storedHeartbeat) == "not-a-date-time" ||
  //     (currentTime - storedHeartbeat).total_milliseconds() > fHeartbeatTimeoutInMs) {
  //   LOG(WARN) << "Heartbeat too old for EPN#" << direction << ", discarding message.";
  //   fSTFBuffer.pop();
  //   fArrivalTime.pop();
  // } else { // if the heartbeat from the corresponding EPN is within timeout period, send the data.
    if (SendAsync(fSTFBuffer.front(), "data-out", direction) < 0) {
      LOG(ERROR) << "Failed to queue sub-timeframe #" << currentTimeframeId;
    }
    fSTFBuffer.pop();
    fArrivalTime.pop();
  // }
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
