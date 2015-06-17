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

using namespace std;

O2EPNex::O2EPNex() :
  fHeartbeatIntervalInMs(5000)
{
}

O2EPNex::~O2EPNex()
{
}

void O2EPNex::Run()
{
  boost::posix_time::ptime referenceTime = boost::posix_time::microsec_clock::local_time();

  // Set the time difference to fHeartbeatIntervalInMs to immediately send a heartbeat to the EPNs
  int timeDif = fHeartbeatIntervalInMs;
  string ownAddress = fChannels["data-in"].at(0).GetAddress();
  int ownAddressLength = strlen(ownAddress.c_str());

  while (GetCurrentState() == RUNNING) {
    if (timeDif >= fHeartbeatIntervalInMs) {
      referenceTime = boost::posix_time::microsec_clock::local_time();

      for (int i = 0; i < fChannels["data-out"].size(); ++i) {
        FairMQMessage* heartbeatMsg = fTransportFactory->CreateMessage(ownAddressLength);
        memcpy(heartbeatMsg->GetData(), ownAddress.c_str(), ownAddressLength);

        fChannels["data-out"].at(i).Send(heartbeatMsg);

        delete heartbeatMsg;
      }
    }

    // Update the time difference
    timeDif = (boost::posix_time::microsec_clock::local_time() - referenceTime).total_milliseconds();

    // Receive payload
    FairMQMessage* payloadMsg = fTransportFactory->CreateMessage();

    if (fChannels["data-in"].at(0).Receive(payloadMsg, "no-block") > 0) {
      int inputSize = payloadMsg->GetSize();
      int numInput = inputSize / sizeof(Content);
      Content* input = reinterpret_cast<Content*>(payloadMsg->GetData());

      // for (int i = 0; i < numInput; ++i) {
      //     LOG(INFO) << (&input[i])->x << " " << (&input[i])->y << " " << (&input[i])->z << " " << (&input[i])->a << " " << (&input[i])->b;
      // }
    }

    delete payloadMsg;
  }
}

void O2EPNex::SetProperty(const int key, const string& value)
{
  switch (key) {
    default:
      FairMQDevice::SetProperty(key, value);
      break;
  }
}

string O2EPNex::GetProperty(const int key, const string& default_/*= ""*/)
{
  switch (key) {
    default:
      return FairMQDevice::GetProperty(key, default_);
  }
}

void O2EPNex::SetProperty(const int key, const int value)
{
  switch (key) {
    case HeartbeatIntervalInMs:
      fHeartbeatIntervalInMs = value;
      break;
    default:
      FairMQDevice::SetProperty(key, value);
      break;
  }
}

int O2EPNex::GetProperty(const int key, const int default_/*= 0*/)
{
  switch (key) {
    case HeartbeatIntervalInMs:
      return fHeartbeatIntervalInMs;
    default:
      return FairMQDevice::GetProperty(key, default_);
  }
}
