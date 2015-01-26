/**
 * EPNex.cxx
 *
 * @since 2013-01-09
 * @author D. Klein, A. Rybalchenko, M. Al-Turany, C. Kouzinopoulos
 */

#include <boost/thread.hpp>
#include <boost/bind.hpp>
#include <fstream> // writing to file (DEBUG)

#include "EPNex.h"
#include "FairMQLogger.h"

using namespace std;

using namespace AliceO2::Devices;

struct f2eHeader {
  uint64_t timeFrameId;
  int      flpId;
};

EPNex::EPNex()
  : fHeartbeatIntervalInMs(3000)
  , fBufferTimeoutInMs(1000)
  , fNumFLPs(1)
  , fTimeframeBuffer()
  , fDiscardedSet()
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

void EPNex::DiscardIncompleteTimeframes()
{
  unordered_map<uint64_t,timeframeBuffer>::iterator it = fTimeframeBuffer.begin();
  while (it != fTimeframeBuffer.end()) {
    if ((boost::posix_time::microsec_clock::local_time() - (it->second).startTime).total_milliseconds() > fBufferTimeoutInMs) {
      LOG(WARN) << "Timeframe #" << it->first << " incomplete after " << fBufferTimeoutInMs << " milliseconds, discarding";
      fDiscardedSet.insert(it->first);
      for(int i = 0; i < (it->second).parts.size(); ++i) {
        delete (it->second).parts.at(i);
      }
      it->second.parts.clear();
      fTimeframeBuffer.erase(it++);
      LOG(WARN) << "Number of discarded timeframes: " << fDiscardedSet.size();
    } else {
      // LOG(INFO) << "Timeframe #" << it->first << " within timeout, buffering...";
      ++it;
    }
  }
}

void EPNex::Run()
{
  LOG(INFO) << ">>>>>>> Run <<<<<<<";

  boost::thread rateLogger(boost::bind(&FairMQDevice::LogSocketRates, this));
  boost::thread heartbeatSender(boost::bind(&EPNex::sendHeartbeats, this));

  int SNDMORE = fPayloadInputs->at(0)->SNDMORE;
  int NOBLOCK = fPayloadInputs->at(0)->NOBLOCK;

  // DEBUG: store receive intervals per FLP
  vector<vector<int>> rcvIntervals(fNumFLPs, vector<int>());
  vector<boost::posix_time::ptime> rcvTimestamp(fNumFLPs);
  // end DEBUG

  f2eHeader* h; // holds the header of the currently arrived message.
  uint64_t id = 0; // holds the timeframe id of the currently arrived sub-timeframe.

  while (fState == RUNNING) {
    FairMQMessage* headerPart = fTransportFactory->CreateMessage();

    if (fPayloadInputs->at(0)->Receive(headerPart) > 0) {
      // store the received ID
      h = reinterpret_cast<f2eHeader*>(headerPart->GetData());
      id = h->timeFrameId;
      // LOG(INFO) << "Received Timeframe #" << id << " from FLP" << h->flpId;

      // DEBUG:: store receive intervals per FLP
      int flp_id = h->flpId - 1; // super dirty temporary hack 
      if (to_simple_string(rcvTimestamp.at(flp_id)) != "not_a_date_time") {
        rcvIntervals.at(flp_id).push_back( (boost::posix_time::microsec_clock::local_time() - rcvTimestamp.at(flp_id)).total_microseconds() );
        // LOG(WARN) << rcvIntervals.at(flp_id).back();
      }
      rcvTimestamp.at(flp_id) = boost::posix_time::microsec_clock::local_time();
      // end DEBUG

      if (fDiscardedSet.find(id) == fDiscardedSet.end() && fTimeframeBuffer.find(id) == fTimeframeBuffer.end()) {
        // if received ID is not yet in the buffer and has not previously been discarded.
        FairMQMessage* dataPart = fTransportFactory->CreateMessage();
        if (fPayloadInputs->at(0)->Receive(dataPart) > 0) {
          // receive data, store it in the buffer, save the receive time.
          fTimeframeBuffer[id].count = 1;
          fTimeframeBuffer[id].parts.push_back(dataPart);
          fTimeframeBuffer[id].startTime = boost::posix_time::microsec_clock::local_time();
        } else {
          LOG(ERROR) << "no data received from input socket";
          delete dataPart;
        }
        // PrintBuffer(fTimeframeBuffer);
      } else {
        // if received ID is already in the buffer
        FairMQMessage* dataPart = fTransportFactory->CreateMessage();
        if (fPayloadInputs->at(0)->Receive(dataPart) > 0) {
          fTimeframeBuffer[id].count++;
          fTimeframeBuffer[id].parts.push_back(dataPart);
        } else {
          LOG(ERROR) << "no data received from input socket 0";
          delete dataPart;
        }
        // PrintBuffer(fTimeframeBuffer);
      }

      if (fTimeframeBuffer[id].count == fNumFLPs) {
        // when all parts are collected send all except last one with 'snd-more' flag, and last one without the flag.
        for (int i = 0; i < fNumFLPs - 1; ++i) {
          fPayloadOutputs->at(fNumFLPs)->Send(fTimeframeBuffer[id].parts.at(i), SNDMORE);
        }
        fPayloadOutputs->at(fNumFLPs)->Send(fTimeframeBuffer[id].parts.at(fNumFLPs - 1));

        // Send an acknowledgement back to the sampler to measure the round trip time
        FairMQMessage* ack = fTransportFactory->CreateMessage(sizeof(uint64_t));
        memcpy(ack->GetData(), &id, sizeof(uint64_t));

        if (fPayloadOutputs->at(fNumFLPs + 1)->Send(ack, NOBLOCK) == 0) {
          LOG(ERROR) << "Could not send acknowledgement without blocking";
        }

        delete ack;

        // let transport know that the data is no longer needed. transport will clean up after it is sent out.
        for(int i = 0; i < fTimeframeBuffer[id].parts.size(); ++i) {
          delete fTimeframeBuffer[id].parts.at(i);
        }
        fTimeframeBuffer[id].parts.clear();

        // fTimeframeBuffer[id].endTime = boost::posix_time::microsec_clock::local_time();
        // do something with time here ...
        fTimeframeBuffer.erase(id);
      }

      // check if any incomplete timeframes in the buffer are older than timeout period, and discard them if they are
      DiscardIncompleteTimeframes();

      // LOG(WARN) << "Buffer size: " << fTimeframeBuffer.size();
    }
    delete headerPart;
  }

  // DEBUG: save 
  string name = to_iso_string(boost::posix_time::microsec_clock::local_time()).substr(0, 20);
  for (int x = 0; x < fNumFLPs; ++x) {
    ofstream flpRcvTimes(fId + "-" + name + "-flp-" + to_string(x) + ".log");
    for (auto it = rcvIntervals.at(x).begin() ; it != rcvIntervals.at(x).end(); ++it) {
      flpRcvTimes << *it << endl;
    }
    flpRcvTimes.close();
  }
  // end DEBUG

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
