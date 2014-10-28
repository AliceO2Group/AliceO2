/**
 * EPNex.cxx
 *
 * @since 2013-01-09
 * @author D. Klein, A. Rybalchenko, M.Al-Turany, C. Kouzinopoulos
 */

#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/thread.hpp>
#include <boost/bind.hpp>

#include "EPNex.h"
#include "FairMQLogger.h"

using namespace std;

using namespace AliceO2::Devices;

EPNex::EPNex() :
  fHeartbeatIntervalInMs(5000)
{
}

EPNex::~EPNex()
{
}

void EPNex::Run()
{
  LOG(INFO) << ">>>>>>> Run <<<<<<<";

  // boost::thread rateLogger(boost::bind(&FairMQDevice::LogSocketRates, this));
  boost::thread heartbeatSender(boost::bind(&EPNex::sendHeartbeats, this));

  size_t idPartSize = 0;
  size_t dataPartSize = 0;

  while (fState == RUNNING) {
    // Receive payload
    FairMQMessage* idPart = fTransportFactory->CreateMessage();

    idPartSize = fPayloadInputs->at(0)->Receive(idPart);

    if (idPartSize > 0) {
      unsigned long* id = reinterpret_cast<unsigned long*>(idPart->GetData());
      LOG(INFO) << "Received Event #" << *id;

      FairMQMessage* dataPart = fTransportFactory->CreateMessage();
      dataPartSize = fPayloadInputs->at(0)->Receive(dataPart);

      if (dataPartSize > 0) {
        // ... do something with data here ...
      }
      delete dataPart;
    }
    delete idPart;
  }

  // rateLogger.interrupt();
  // rateLogger.join();

  heartbeatSender.interrupt();
  heartbeatSender.join();

  FairMQDevice::Shutdown();

  // notify parent thread about end of processing.
  boost::lock_guard<boost::mutex> lock(fRunningMutex);
  fRunningFinished = true;
  fRunningCondition.notify_one();
}

void EPNex::sendHeartbeats()
{
  while (true) {
    try {
      for (int i = 0; i < fNumOutputs; ++i) {
        FairMQMessage* heartbeatMsg = fTransportFactory->CreateMessage(fInputAddress.at(0).size());
        memcpy(heartbeatMsg->GetData(), fInputAddress.at(0).c_str(), fInputAddress.at(0).size());

        fPayloadOutputs->at(i)->Send(heartbeatMsg);

        delete heartbeatMsg;
      }
      boost::this_thread::sleep(boost::posix_time::milliseconds(fHeartbeatIntervalInMs));
    } catch (boost::thread_interrupted&) {
      LOG(INFO) << "EPNex::sendHeartbeat() interrupted";
      break;
    }
  } // while (true)
}

void EPNex::SetProperty(const int key, const string& value, const int slot/*= 0*/)
{
  switch (key) {
    default:
      FairMQDevice::SetProperty(key, value, slot);
      break;
  }
}

string EPNex::GetProperty(const int key, const string& default_/*= ""*/, const int slot/*= 0*/)
{
  switch (key) {
    default:
      return FairMQDevice::GetProperty(key, default_, slot);
  }
}

void EPNex::SetProperty(const int key, const int value, const int slot/*= 0*/)
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

int EPNex::GetProperty(const int key, const int default_/*= 0*/, const int slot/*= 0*/)
{
  switch (key) {
    case HeartbeatIntervalInMs:
      return fHeartbeatIntervalInMs;
    default:
      return FairMQDevice::GetProperty(key, default_, slot);
  }
}
