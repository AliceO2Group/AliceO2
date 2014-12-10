/**
 * EPNex.cxx
 *
 * @since 2013-01-09
 * @author D. Klein, A. Rybalchenko, M. Al-Turany, C. Kouzinopoulos
 */

#include <fstream>

#include <boost/thread.hpp>
#include <boost/bind.hpp>

#include "EPNex.h"
#include "FairMQLogger.h"

using namespace std;

using namespace AliceO2::Devices;

EPNex::EPNex() :
  fHeartbeatIntervalInMs(3000),
  fNumFLPs(1)
{
}

EPNex::~EPNex()
{
}

void EPNex::PrintBuffer(unordered_map<uint64_t,timeframeBuffer> &buffer)
{
  int size = buffer.size();
  string header = "===== ";

  for (int i = 1; i <= fNumFLPs; ++i) {
    stringstream out;
    out << i % 10;
    header += out.str();
    //i > 9 ? header += " " : header += "  ";
  }
  LOG(INFO) << header;

  for (unordered_map<uint64_t,timeframeBuffer>::iterator it = buffer.begin(); it != buffer.end(); ++it) {
    string stars = "";
    for (int j = 1; j <= (it->second).count; ++j) {
      stars += "*";
    }
    LOG(INFO) << setw(4) << it->first << ": " << stars;
  }
}

void EPNex::Run()
{
  LOG(INFO) << ">>>>>>> Run <<<<<<<";

  boost::thread rateLogger(boost::bind(&FairMQDevice::LogSocketRates, this));
  boost::thread heartbeatSender(boost::bind(&EPNex::sendHeartbeats, this));

  uint64_t id = 0; // holds the timeframe id of the currently arrived timeframe.
  fNumOfDiscardedTimeframes = 0;

  while (fState == RUNNING) {
    FairMQMessage* idPart = fTransportFactory->CreateMessage();

    if (fPayloadInputs->at(0)->Receive(idPart) > 0) {
      // store the received ID
      id = *(reinterpret_cast<uint64_t*>(idPart->GetData()));
      // LOG(INFO) << "Received Timeframe #" << id;

      if (fTimeframeBuffer.find(id) == fTimeframeBuffer.end()) {
        // if received ID is not yet in the map
        FairMQMessage* dataPart = fTransportFactory->CreateMessage();
        if (fPayloadInputs->at(0)->Receive(dataPart) > 0) {
          // receive data, store it in the buffer, save the receive time.
          fTimeframeBuffer[id].count = 1;
          fTimeframeBuffer[id].parts.push_back(dataPart);
          fTimeframeBuffer[id].startTime = boost::posix_time::microsec_clock::local_time();
        } else {
          LOG(ERROR) << "no data received from input socket 0";
          delete dataPart;
        }
        // PrintBuffer(fTimeframeBuffer);
      } else {
        // if received ID is already in the map
        FairMQMessage* dataPart = fTransportFactory->CreateMessage();
        if (fPayloadInputs->at(0)->Receive(dataPart) > 0) {
          fTimeframeBuffer[id].count++;
          fTimeframeBuffer[id].parts.push_back(dataPart);
        } else {
          LOG(ERROR) << "no data received from input socket 0";
          delete dataPart;
        }
        // PrintBuffer(fTimeframeBuffer);
        if (fTimeframeBuffer[id].count == fNumFLPs) {
          // when all parts are collected send all except last one with 'snd-more' flag, and last one without the flag.
          for (int i = 0; i < fNumFLPs - 1; ++i) {
            fPayloadOutputs->at(fNumFLPs)->Send(fTimeframeBuffer[id].parts.at(i), "snd-more");
          }
          fPayloadOutputs->at(fNumFLPs)->Send(fTimeframeBuffer[id].parts.at(fNumFLPs - 1));

          // let transport know that the data is no longer needed. transport will clean up when it is out.
          for(int i = 0; i < fTimeframeBuffer[id].parts.size(); ++i) {
            delete fTimeframeBuffer[id].parts.at(i);
          }
          fTimeframeBuffer[id].parts.clear();

          fTimeframeBuffer[id].endTime = boost::posix_time::microsec_clock::local_time();
          // do something with time here ...
          fTimeframeBuffer.erase(id);
        }
      }

      // check if any incomplete timeframes in the buffer are older than timeout period, and discard them
      unordered_map<uint64_t,timeframeBuffer>::iterator it = fTimeframeBuffer.begin();
      while (it != fTimeframeBuffer.end()) {
        if ((boost::posix_time::microsec_clock::local_time() - (it->second).startTime).total_milliseconds() > fBufferTimeoutInMs) {
          LOG(WARN) << "Timeframe #" << it->first << " incomplete after " << fBufferTimeoutInMs << " milliseconds, discarding";
          for(int i = 0; i < (it->second).parts.size(); ++i) {
            delete (it->second).parts.at(i);
          }
          it->second.parts.clear();
          fTimeframeBuffer.erase(it++);
          fNumOfDiscardedTimeframes++;
          LOG(WARN) << "Number of discarded timeframes: " << fNumOfDiscardedTimeframes;
        } else {
          // LOG(INFO) << "Timeframe #" << it->first << " within timeout, buffering...";
          ++it;
        }
      }

      // LOG(WARN) << "Buffer size: " << fTimeframeBuffer.size();
    }
    delete idPart;
  }

  // std::ofstream ofs(fId + "times.log");
  // for (unordered_map<uint64_t,timeframeDuration>::iterator it = fFullTimeframeBuffer.begin(); it != fFullTimeframeBuffer.end(); ++it) {
  //   ofs << it->first << ": " << ((it->second).end - (it->second).start).total_milliseconds() << "\n";
  // }
  // ofs.close();

  rateLogger.interrupt();
  rateLogger.join();

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
      for (int i = 0; i < fNumFLPs; ++i) {
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
    case BufferTimeoutInMs:
      fBufferTimeoutInMs = value;
      break;
    case NumFLPs:
      fNumFLPs = value;
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
    case BufferTimeoutInMs:
      return fBufferTimeoutInMs;
    case NumFLPs:
      return fNumFLPs;
    default:
      return FairMQDevice::GetProperty(key, default_, slot);
  }
}
