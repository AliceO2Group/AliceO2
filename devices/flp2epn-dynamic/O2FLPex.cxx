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

#include "O2FLPex.h"
#include "FairMQLogger.h"

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
  
  for (int i = 0; i < fNumOutputs; ++i) {
    fOutputHeartbeat.push_back(nullTime);
  }
}

bool O2FLPex::updateIPHeartbeat (string str)
{
  for (int iOutput = 0; iOutput < fNumOutputs; iOutput++) {
    if ( GetProperty (OutputAddress, "", iOutput) == str ) {
      boost::posix_time::ptime currentHeartbeat = boost::posix_time::microsec_clock::local_time();
      boost::posix_time::ptime storedHeartbeat = GetProperty (OutputHeartbeat, storedHeartbeat, iOutput);

      //if ( to_simple_string (storedHeartbeat) != "not-a-date-time" ) {
      //  LOG(INFO) << "EPN " << iOutput << " last seen "
      //            << (currentHeartbeat - storedHeartbeat).total_milliseconds()
      //            << " ms ago. Updating heartbeat...";
      //}
      //else {
      //  LOG(INFO) << "IP has no heartbeat associated. Adding heartbeat: " << currentHeartbeat;
      //}

      SetProperty (OutputHeartbeat, currentHeartbeat, iOutput);
      
      return true;
    }
  }
  LOG(ERROR) << "IP " << str << " unknown, not provided at execution time";

  return false;
}

void O2FLPex::Run()
{
  LOG(INFO) << ">>>>>>> Run <<<<<<<";

  boost::thread rateLogger (boost::bind(&FairMQDevice::LogSocketRates, this));

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
        //LOG(INFO) << (&payload[i])->id << " " << (&payload[i])->x << " " << (&payload[i])->y << " " << (&payload[i])->z << " " << (&payload[i])->a << " " << (&payload[i])->b;
  }

  delete[] payload;
  
  while ( fState == RUNNING ) {
    //Receive heartbeat
    FairMQMessage* heartbeatMsg = fTransportFactory->CreateMessage();

    size_t heartbeatSize = fPayloadInputs->at(0)->Receive(heartbeatMsg, "no-block");
    
    if ( heartbeatSize > 0 ) {
      std::string rpl = std::string (static_cast<char*>(heartbeatMsg->GetData()), heartbeatMsg->GetSize());
      updateIPHeartbeat (rpl);
    }
    
    delete heartbeatMsg;
    
    //Send payload
    for (int iOutput = 0; iOutput < fNumOutputs; iOutput++) {
      boost::posix_time::ptime currentHeartbeat = boost::posix_time::microsec_clock::local_time();
      boost::posix_time::ptime storedHeartbeat = GetProperty (OutputHeartbeat, storedHeartbeat, iOutput);
      
      if ( to_simple_string (storedHeartbeat) == "not-a-date-time" ||
         (currentHeartbeat - storedHeartbeat).total_milliseconds() > fHeartbeatTimeoutInMs) {
        //LOG(INFO) << "EPN " << iOutput << " has not send a heartbeat, or heartbeat too old";
        continue;
      }
      
      //LOG(INFO) << "Pubishing payload to EPN " << iOutput;
      FairMQMessage* payloadMsg = fTransportFactory->CreateMessage(fEventSize * sizeof(Content));
      memcpy(payloadMsg->GetData(), payload, fEventSize * sizeof(Content));
      
      fPayloadOutputs->at(iOutput)->Send(payloadMsg);
      
      delete payloadMsg;
    }
  }
  
  rateLogger.interrupt();

  rateLogger.join();

  FairMQDevice::Shutdown();

  // notify parent thread about end of processing.
  boost::lock_guard<boost::mutex> lock(fRunningMutex);
  fRunningFinished = true;
  fRunningCondition.notify_one();
}

void O2FLPex::Log(int intervalInMs)
{
  timestamp_t t0;
  timestamp_t t1;
  unsigned long bytes = fPayloadOutputs->at(0)->GetBytesTx();
  unsigned long messages = fPayloadOutputs->at(0)->GetMessagesTx();
  unsigned long bytesNew = 0;
  unsigned long messagesNew = 0;
  double megabytesPerSecond = 0;
  double messagesPerSecond = 0;

  t0 = get_timestamp();

  while (true) {
    boost::this_thread::sleep(boost::posix_time::milliseconds(intervalInMs));

    t1 = get_timestamp();

    bytesNew = fPayloadOutputs->at(0)->GetBytesTx();
    messagesNew = fPayloadOutputs->at(0)->GetMessagesTx();

    timestamp_t timeSinceLastLog_ms = (t1 - t0) / 1000.0L;

    megabytesPerSecond = ((double) (bytesNew - bytes) / (1024. * 1024.)) / (double) timeSinceLastLog_ms * 1000.;
    messagesPerSecond = (double) (messagesNew - messages) / (double) timeSinceLastLog_ms * 1000.;

    LOG(DEBUG) << "send " << messagesPerSecond << " msg/s, " << megabytesPerSecond << " MB/s";

    bytes = bytesNew;
    messages = messagesNew;
    t0 = t1;
  }
}

void O2FLPex::SetProperty(const int key, const string& value, const int slot/*= 0*/)
{
  switch (key) {
  default:
    FairMQDevice::SetProperty(key, value, slot);
    break;
  }
}

string O2FLPex::GetProperty(const int key, const string& default_/*= ""*/, const int slot/*= 0*/)
{
  switch (key) {
  default:
    return FairMQDevice::GetProperty(key, default_, slot);
  }
}

void O2FLPex::SetProperty(const int key, const int value, const int slot/*= 0*/)
{
  switch (key) {
  case EventSize:
    fEventSize = value;
    break;
  case HeartbeatTimeoutInMs:
    fHeartbeatTimeoutInMs = value;
    break;
  default:
    FairMQDevice::SetProperty(key, value, slot);
    break;
  }
}

int O2FLPex::GetProperty(const int key, const int default_/*= 0*/, const int slot/*= 0*/)
{
  switch (key) {
  case EventSize:
    return fEventSize;
  case HeartbeatTimeoutInMs:
    return fHeartbeatTimeoutInMs;
  default:
    return FairMQDevice::GetProperty(key, default_, slot);
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
}
