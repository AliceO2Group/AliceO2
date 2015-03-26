/**
 * FLPSender.cxx
 *
 * @since 2013-04-23
 * @author D. Klein, A. Rybalchenko, M. Al-Turany, C. Kouzinopoulos
 */

#include <vector>
#include <cstdint> // UINT64_MAX

#include <boost/thread.hpp>
#include <boost/bind.hpp>

#include "FairMQLogger.h"
#include "FairMQPoller.h"

#include "FLPSender.h"

using namespace std;
using boost::posix_time::ptime;

using namespace AliceO2::Devices;

struct f2eHeader {
  uint64_t timeFrameId;
  int      flpId;
};

FLPSender::FLPSender()
  : fHeartbeatTimeoutInMs(20000)
  , fOutputHeartbeat()
  , fSendOffset(0)
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
  FairMQDevice::Init();

  ptime nullTime;

  for (int i = 0; i < fNumOutputs; ++i) {
    fOutputHeartbeat.push_back(nullTime);
  }
}

bool FLPSender::updateIPHeartbeat(string reply)
{
  for (int i = 0; i < fNumOutputs; ++i) {
    if (GetProperty(OutputAddress, "", i) == reply) {
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
  LOG(INFO) << ">>>>>>> Run <<<<<<<";

  boost::thread rateLogger(boost::bind(&FairMQDevice::LogSocketRates, this));

  FairMQPoller* poller = fTransportFactory->CreatePoller(*fPayloadInputs);

  // base buffer, to be copied from for every timeframe body
  void* buffer = operator new[](fEventSize);
  FairMQMessage* baseMsg = fTransportFactory->CreateMessage(buffer, fEventSize);

  ptime currentTime;
  ptime storedHeartbeat;

  uint64_t timeFrameId = 0;

  while (fState == RUNNING) {
    poller->Poll(2);

    // input 0 - commands
    if (poller->CheckInput(0)) {
      FairMQMessage* commandMsg = fTransportFactory->CreateMessage();

      if (fPayloadInputs->at(0)->Receive(commandMsg) > 0) {
        //... handle command ...
      }

      delete commandMsg;
    }

    // input 1 - heartbeats
    if (poller->CheckInput(1)) {
      FairMQMessage* heartbeatMsg = fTransportFactory->CreateMessage();

      if (fPayloadInputs->at(1)->Receive(heartbeatMsg) > 0) {
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
        fPayloadInputs->at(2)->Receive(idPart);

        h->timeFrameId = *(reinterpret_cast<uint64_t*>(idPart->GetData()));
        h->flpId = stoi(fId);

        delete idPart;
      } else {
        // regular mode: use the id generated locally
        h->timeFrameId = timeFrameId;
        // h->flpId = stoi(fId);
        h->flpId = 0;

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
        fPayloadInputs->at(2)->Receive(dataPart);
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
      if ((now - fArrivalTime.front()).total_milliseconds() >= (8 * fSendOffset)) {
        sendFrontData();
      } else {
        // LOG(INFO) << "buffering...";
      }
    }
  }

  delete baseMsg;

  rateLogger.interrupt();
  rateLogger.join();

  FairMQDevice::Shutdown();

  // notify parent thread about end of processing.
  boost::lock_guard<boost::mutex> lock(fRunningMutex);
  fRunningFinished = true;
  fRunningCondition.notify_one();
}

inline void FLPSender::sendFrontData()
{
  f2eHeader h = *(reinterpret_cast<f2eHeader*>(fHeaderBuffer.front()->GetData()));
  uint64_t currentTimeframeId = h.timeFrameId;

  int SNDMORE = fPayloadInputs->at(0)->SNDMORE;
  int NOBLOCK = fPayloadInputs->at(0)->NOBLOCK;

  // for which EPN is the message?
  int direction = currentTimeframeId % fNumOutputs;
  // LOG(INFO) << "Sending event " << currentTimeframeId << " to EPN#" << direction << "...";

  ptime currentTime = boost::posix_time::microsec_clock::local_time();
  ptime storedHeartbeat = GetProperty(OutputHeartbeat, storedHeartbeat, direction);

  // if the heartbeat from the corresponding EPN is within timeout period, send the data.
  if (to_simple_string(storedHeartbeat) != "not-a-date-time" ||
      (currentTime - storedHeartbeat).total_milliseconds() < fHeartbeatTimeoutInMs) {
    if(fPayloadOutputs->at(direction)->Send(fHeaderBuffer.front(), SNDMORE|NOBLOCK) == 0) {
      LOG(ERROR) << "Could not queue ID part of event #" << currentTimeframeId << " without blocking";
    }
    if (fPayloadOutputs->at(direction)->Send(fDataBuffer.front(), NOBLOCK) == 0) {
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

void FLPSender::SetProperty(const int key, const string& value, const int slot/*= 0*/)
{
  switch (key) {
    default:
      FairMQDevice::SetProperty(key, value, slot);
      break;
  }
}

string FLPSender::GetProperty(const int key, const string& default_/*= ""*/, const int slot/*= 0*/)
{
  switch (key) {
    default:
      return FairMQDevice::GetProperty(key, default_, slot);
  }
}

void FLPSender::SetProperty(const int key, const int value, const int slot/*= 0*/)
{
  switch (key) {
    case HeartbeatTimeoutInMs:
      fHeartbeatTimeoutInMs = value;
      break;
    case TestMode:
      fTestMode = value;
      break;
    case SendOffset:
      fSendOffset = value;
      break;
    case EventSize:
      fEventSize = value;
      break;
    default:
      FairMQDevice::SetProperty(key, value, slot);
      break;
  }
}

int FLPSender::GetProperty(const int key, const int default_/*= 0*/, const int slot/*= 0*/)
{
  switch (key) {
    case HeartbeatTimeoutInMs:
      return fHeartbeatTimeoutInMs;
    case TestMode:
      return fTestMode;
    case SendOffset:
      return fSendOffset;
    case EventSize:
      return fEventSize;
    default:
      return FairMQDevice::GetProperty(key, default_, slot);
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
}
