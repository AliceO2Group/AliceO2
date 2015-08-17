/**
 * O2FLPex.cxx
 *
 * @since 2013-04-23
 * @author D. Klein, A. Rybalchenko, M.Al-Turany, C. Kouzinopoulos
 */

#include <vector>
#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */

#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/thread.hpp>
#include <boost/bind.hpp>
#include <iostream>
#include <sstream>
#include <cassert>

#include "O2FLPex.h"
#include "FairMQLogger.h"

using namespace std;

O2FLPex::O2FLPex() :
  fEventSize(10000),
  fHeartbeatTimeoutInMs(20000)
{
}

O2FLPex::~O2FLPex()
{
}

void O2FLPex::Init()
{
  FairMQDevice::Init();

  boost::posix_time::ptime nullTime;

  for (int i = 0; i < fChannels["data-out"].size(); ++i) {
    fOutputHeartbeat.push_back(nullTime);
  }
}

bool O2FLPex::updateIPHeartbeat (string str)
{
  for (int i = 0; i < fChannels["data-out"].size(); ++i) {
    if ( fChannels["data-out"].at(i).GetAddress() == str ) {
      boost::posix_time::ptime currentHeartbeat = boost::posix_time::microsec_clock::local_time();
      boost::posix_time::ptime storedHeartbeat = GetProperty (OutputHeartbeat, storedHeartbeat, i);

      if (to_simple_string(storedHeartbeat) != "not-a-date-time") {
        LOG(INFO) << "EPN " << i << " (" << str << ")" << " last seen "
                  << (currentHeartbeat - storedHeartbeat).total_milliseconds()
                  << " ms ago. Updating heartbeat...";
      }
      else {
        LOG(INFO) << "IP has no heartbeat associated. Adding heartbeat: " << currentHeartbeat;
      }

      SetProperty (OutputHeartbeat, currentHeartbeat, i);

      return true;
    }
  }
  LOG(ERROR) << "IP " << str << " unknown, not provided at execution time";

  return false;
}

void O2FLPex::Run()
{
  srand(time(NULL));
 
  stringstream ss(fId);

  int Flp_id;
  ss >> Flp_id;

  Content* payload = new Content[fEventSize];
  for (int i = 0; i < fEventSize; ++i) {
        (&payload[i])->id = Flp_id;
        (&payload[i])->x = rand() % 100 + 1;
        (&payload[i])->y = rand() % 100 + 1;
        (&payload[i])->z = rand() % 100 + 1;
        (&payload[i])->a = (rand() % 100 + 1) / (rand() % 100 + 1);
        (&payload[i])->b = (rand() % 100 + 1) / (rand() % 100 + 1);
        // LOG(INFO) << (&payload[i])->id << " " << (&payload[i])->x << " " << (&payload[i])->y << " " << (&payload[i])->z << " " << (&payload[i])->a << " " << (&payload[i])->b;
  }

  delete[] payload;

  while (GetCurrentState() == RUNNING) {
    // Receive heartbeat
    FairMQMessage* heartbeatMsg = fTransportFactory->CreateMessage();

    size_t heartbeatSize = fChannels["data-in"].at(0).Receive(heartbeatMsg, "no-block");

    if (heartbeatSize > 0) {
      std::string rpl = std::string (static_cast<char*>(heartbeatMsg->GetData()), heartbeatMsg->GetSize());
      updateIPHeartbeat (rpl);
    }

    delete heartbeatMsg;

    // Send payload
    for (int i = 0; i < fChannels["data-out"].size(); ++i) {
      boost::posix_time::ptime currentHeartbeat = boost::posix_time::microsec_clock::local_time();
      boost::posix_time::ptime storedHeartbeat = GetProperty (OutputHeartbeat, storedHeartbeat, i);

      if (to_simple_string(storedHeartbeat) == "not-a-date-time" ||
         (currentHeartbeat - storedHeartbeat).total_milliseconds() > fHeartbeatTimeoutInMs) {
        // LOG(INFO) << "EPN " << i << " has not send a heartbeat, or heartbeat too old";
        continue;
      }

      // LOG(INFO) << "Pubishing payload to EPN " << i;
      FairMQMessage* payloadMsg = fTransportFactory->CreateMessage(fEventSize * sizeof(Content));
      memcpy(payloadMsg->GetData(), payload, fEventSize * sizeof(Content));

      fChannels["data-out"].at(i).Send(payloadMsg);

      delete payloadMsg;
    }
  }
}

void O2FLPex::SetProperty(const int key, const string& value)
{
  switch (key) {
    default:
      FairMQDevice::SetProperty(key, value);
      break;
  }
}

string O2FLPex::GetProperty(const int key, const string& default_/*= ""*/)
{
  switch (key) {
    default:
      return FairMQDevice::GetProperty(key, default_);
  }
}

void O2FLPex::SetProperty(const int key, const int value)
{
  switch (key) {
    case EventSize:
      fEventSize = value;
      break;
    case HeartbeatTimeoutInMs:
      fHeartbeatTimeoutInMs = value;
      break;
    default:
      FairMQDevice::SetProperty(key, value);
      break;
  }
}

int O2FLPex::GetProperty(const int key, const int default_/*= 0*/)
{
  switch (key) {
    case EventSize:
      return fEventSize;
    case HeartbeatTimeoutInMs:
      return fHeartbeatTimeoutInMs;
    default:
      return FairMQDevice::GetProperty(key, default_);
  }
}

// Method for setting properties represented as a heartbeat.
void O2FLPex::SetProperty(const int key, const boost::posix_time::ptime value, const int slot /*= 0*/)
{
  switch (key)
  {
    case OutputHeartbeat:
      fOutputHeartbeat.erase(fOutputHeartbeat.begin() + slot);
      fOutputHeartbeat.insert(fOutputHeartbeat.begin() + slot, value);
      break;
  }
}

// Method for getting properties represented as a heartbeat.
boost::posix_time::ptime O2FLPex::GetProperty(const int key, const boost::posix_time::ptime default_, const int slot /*= 0*/)
{
  switch (key)
  {
    case OutputHeartbeat:
      return fOutputHeartbeat.at(slot);
  }
  assert(false);
}
