/**
 * O2EPNex.cxx
 *
 * @since 2013-01-09
 * @author D. Klein, A. Rybalchenko, M.Al-Turany, C. Kouzinopoulos
 */

#include <boost/thread.hpp>
#include <boost/bind.hpp>

#include "O2EPNex.h"
#include "FairMQLogger.h"

O2EPNex::O2EPNex() :
  fHeartbeatIntervalInMs(5000)
{
}

O2EPNex::~O2EPNex()
{
}

void O2EPNex::Run()
{
  LOG(INFO) << ">>>>>>> Run <<<<<<<";

  boost::thread rateLogger(boost::bind(&FairMQDevice::LogSocketRates, this));

  boost::posix_time::ptime referenceTime = boost::posix_time::microsec_clock::local_time();

  // Set the time difference to fHeartbeatIntervalInMs to immediately send a heartbeat to the EPNs
  int timeDif = fHeartbeatIntervalInMs;

  while (fState == RUNNING) {
    if (timeDif >= fHeartbeatIntervalInMs) {
      referenceTime = boost::posix_time::microsec_clock::local_time();

      for (int i = 0; i < fNumOutputs; i++) {
        FairMQMessage* heartbeatMsg = fTransportFactory->CreateMessage (strlen (fInputAddress.at(0).c_str()));
        memcpy(heartbeatMsg->GetData(), fInputAddress.at(0).c_str(), strlen (fInputAddress.at(0).c_str()));

        fPayloadOutputs->at(i)->Send(heartbeatMsg);

        delete heartbeatMsg;
      }
    }

    // Update the time difference
    timeDif = (boost::posix_time::microsec_clock::local_time() - referenceTime).total_milliseconds();

    // Receive payload
    FairMQMessage* payloadMsg = fTransportFactory->CreateMessage();

    size_t payloadSize = fPayloadInputs->at(0)->Receive(payloadMsg, "no-block");

    if ( payloadSize > 0 ) {
      int inputSize = payloadMsg->GetSize();
      int numInput = inputSize / sizeof(Content);
      Content* input = reinterpret_cast<Content*>(payloadMsg->GetData());

      // for (int i = 0; i < numInput; ++i) {
      //     LOG(INFO) << (&input[i])->x << " " << (&input[i])->y << " " << (&input[i])->z << " " << (&input[i])->a << " " << (&input[i])->b;
      // }
    }

    delete payloadMsg;
  }

  rateLogger.interrupt();
  rateLogger.join();

  FairMQDevice::Shutdown();

  // notify parent thread about end of processing.
  boost::lock_guard<boost::mutex> lock(fRunningMutex);
  fRunningFinished = true;
  fRunningCondition.notify_one();
}

void O2EPNex::SetProperty(const int key, const string& value, const int slot/*= 0*/)
{
  switch (key) {
    default:
      FairMQDevice::SetProperty(key, value, slot);
      break;
  }
}

string O2EPNex::GetProperty(const int key, const string& default_/*= ""*/, const int slot/*= 0*/)
{
  switch (key) {
    default:
      return FairMQDevice::GetProperty(key, default_, slot);
  }
}

void O2EPNex::SetProperty(const int key, const int value, const int slot/*= 0*/)
{
  switch (key) {
    case HeartbeatIntervalInMs:
      fHeartbeatIntervalInMs = value;
      break;
    default:
      FairMQDevice::SetProperty(key, value, slot);
      break;
  }
}

int O2EPNex::GetProperty(const int key, const int default_/*= 0*/, const int slot/*= 0*/)
{
  switch (key) {
    case HeartbeatIntervalInMs:
      return fHeartbeatIntervalInMs;
    default:
      return FairMQDevice::GetProperty(key, default_, slot);
  }
}
