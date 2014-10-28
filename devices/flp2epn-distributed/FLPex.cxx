/**
 * FLPex.cxx
 *
 * @since 2013-04-23
 * @author D. Klein, A. Rybalchenko, M. Al-Turany, C. Kouzinopoulos
 */

#include <vector>

#include <boost/thread.hpp>
#include <boost/bind.hpp>

#include "FairMQLogger.h"
#include "FairMQPoller.h"

#include "FLPex.h"

using namespace std;
using boost::posix_time::ptime;

using namespace AliceO2::Devices;

FLPex::FLPex()
  : fHeartbeatTimeoutInMs(20000)
  , fSendOffset(0)
{
}

FLPex::~FLPex()
{
}

void FLPex::Init()
{
  FairMQDevice::Init();

  ptime nullTime;

  for (int i = 0; i < fNumOutputs; ++i) {
    fOutputHeartbeat.push_back(nullTime);
  }
}

bool FLPex::updateIPHeartbeat(string reply)
{
  for (int i = 0; i < fNumOutputs; ++i) {
    if (GetProperty(OutputAddress, "", i) == reply) {
      ptime currentHeartbeat = boost::posix_time::microsec_clock::local_time();
      ptime storedHeartbeat = GetProperty(OutputHeartbeat, storedHeartbeat, i);

      if (to_simple_string(storedHeartbeat) != "not-a-date-time") {
        LOG(INFO) << "EPN " << i << " (" << reply << ")" << " last seen "
                  << (currentHeartbeat - storedHeartbeat).total_milliseconds() << " ms ago.";
      }
      else {
        LOG(INFO) << "IP has no heartbeat associated. Adding heartbeat: " << currentHeartbeat;
      }

      SetProperty(OutputHeartbeat, currentHeartbeat, i);

      return true;
    }
  }
  LOG(ERROR) << "IP " << reply << " unknown, not provided at execution time";

  return false;
}

void FLPex::Run()
{
  LOG(INFO) << ">>>>>>> Run <<<<<<<";

  // boost::thread rateLogger(boost::bind(&FairMQDevice::LogSocketRates, this));

  FairMQPoller* poller = fTransportFactory->CreatePoller(*fPayloadInputs);

  unsigned long eventId = 0;
  int direction = 0;
  int counter = 0;
  int sent = 0;
  ptime currentHeartbeat;
  ptime storedHeartbeat;

  while (fState == RUNNING) {
    poller->Poll(100);

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

    // input 2 - data from Sampler
    if (poller->CheckInput(2)) {
      FairMQMessage* idPart = fTransportFactory->CreateMessage();
      if (fPayloadInputs->at(2)->Receive(idPart) > 0) {
        FairMQMessage* dataPart = fTransportFactory->CreateMessage();
        if (fPayloadInputs->at(2)->Receive(dataPart) > 0) {
          unsigned long* id = reinterpret_cast<unsigned long*>(idPart->GetData());
          eventId = *id;
          LOG(INFO) << "Received Event #" << eventId;

          fIdBuffer.push(idPart);
          fDataBuffer.push(dataPart);
        } else {
          LOG(ERROR) << "Could not receive data part.";
          delete dataPart;
          continue;
        }
      } else {
        LOG(ERROR) << "Could not receive id part.";
        delete idPart;
        continue;
      }

      if (counter == fSendOffset) {
        eventId = *(reinterpret_cast<unsigned long*>(fIdBuffer.front()->GetData()));
        direction = eventId % fNumOutputs;

        LOG(INFO) << "Trying to send event " << eventId << " to EPN#" << direction << "...";

        currentHeartbeat = boost::posix_time::microsec_clock::local_time();
        storedHeartbeat = GetProperty(OutputHeartbeat, storedHeartbeat, direction);

        // if the heartbeat from the corresponding EPN is within timeout period, send the data.
        if (to_simple_string(storedHeartbeat) != "not-a-date-time" ||
            (currentHeartbeat - storedHeartbeat).total_milliseconds() < fHeartbeatTimeoutInMs) {
          fPayloadOutputs->at(direction)->Send(fIdBuffer.front(), "snd-more");
          sent = fPayloadOutputs->at(direction)->Send(fDataBuffer.front(), "no-block");
          if (sent == 0) {
            LOG(ERROR) << "Could not send message with event #" << eventId << " without blocking";
          }
          fIdBuffer.pop();
          fDataBuffer.pop();
        } else { // if the heartbeat is too old, receive the data and discard it.
          LOG(WARN) << "Heartbeat too old for, discarding message.";
          fIdBuffer.pop();
          fDataBuffer.pop();
        }
      } else if (counter < fSendOffset) {
        LOG(INFO) << "Buffering event...";
        ++counter;
      } else {
        LOG(ERROR) << "Counter larger than offset, something went wrong...";
      }
    } // if (poller->CheckInput(2))
  } // while (fState == RUNNING)

  // rateLogger.interrupt();
  // rateLogger.join();

  FairMQDevice::Shutdown();

  // notify parent thread about end of processing.
  boost::lock_guard<boost::mutex> lock(fRunningMutex);
  fRunningFinished = true;
  fRunningCondition.notify_one();
}

void FLPex::SetProperty(const int key, const string& value, const int slot/*= 0*/)
{
  switch (key) {
    default:
      FairMQDevice::SetProperty(key, value, slot);
      break;
  }
}

string FLPex::GetProperty(const int key, const string& default_/*= ""*/, const int slot/*= 0*/)
{
  switch (key) {
    default:
      return FairMQDevice::GetProperty(key, default_, slot);
  }
}

void FLPex::SetProperty(const int key, const int value, const int slot/*= 0*/)
{
  switch (key) {
    case HeartbeatTimeoutInMs:
      fHeartbeatTimeoutInMs = value;
      break;
    case SendOffset:
      fSendOffset = value;
      break;
    default:
      FairMQDevice::SetProperty(key, value, slot);
      break;
  }
}

int FLPex::GetProperty(const int key, const int default_/*= 0*/, const int slot/*= 0*/)
{
  switch (key) {
    case HeartbeatTimeoutInMs:
      return fHeartbeatTimeoutInMs;
    case SendOffset:
      return fSendOffset;
    default:
      return FairMQDevice::GetProperty(key, default_, slot);
  }
}

// Method for setting properties represented as a heartbeat.
void FLPex::SetProperty(const int key, const ptime value, const int slot /*= 0*/)
{
  switch (key) {
    case OutputHeartbeat:
      fOutputHeartbeat.erase(fOutputHeartbeat.begin() + slot);
      fOutputHeartbeat.insert(fOutputHeartbeat.begin() + slot, value);
      break;
  }
}

// Method for getting properties represented as a heartbeat.
ptime FLPex::GetProperty(const int key, const ptime default_, const int slot /*= 0*/)
{
  switch (key) {
    case OutputHeartbeat:
      return fOutputHeartbeat.at(slot);
  }
}
