// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/**
 * EPNReceiver.cxx
 *
 * @since 2013-01-09
 * @author D. Klein, A. Rybalchenko, M. Al-Turany, C. Kouzinopoulos
 */

#include <cstddef> // size_t
#include <fstream> // writing to file (DEBUG)
#include <iomanip>

#include <FairMQLogger.h>
#include <options/FairMQProgOptions.h>

#include "FLP2EPNex_distributed/EPNReceiver.h"
#include "O2Device/Compatibility.h"

#include <iomanip>

using namespace std;
using namespace std::chrono;
using namespace o2::devices;

struct f2eHeader {
  uint16_t timeFrameId;
  int flpIndex;
};

EPNReceiver::EPNReceiver()
  : mTimeframeBuffer(), mDiscardedSet(), mNumFLPs(0), mBufferTimeoutInMs(5000), mTestMode(0), mInChannelName(), mOutChannelName(), mAckChannelName()
{
}

EPNReceiver::~EPNReceiver() = default;

void EPNReceiver::InitTask()
{
  mNumFLPs = GetConfig()->GetValue<int>("num-flps");
  mBufferTimeoutInMs = GetConfig()->GetValue<int>("buffer-timeout");
  mTestMode = GetConfig()->GetValue<int>("test-mode");
  mInChannelName = GetConfig()->GetValue<string>("in-chan-name");
  mOutChannelName = GetConfig()->GetValue<string>("out-chan-name");
  mAckChannelName = GetConfig()->GetValue<string>("ack-chan-name");
}

void EPNReceiver::PrintBuffer(const unordered_map<uint16_t, TFBuffer>& buffer) const
{
  string header = "===== ";

  for (int i = 1; i <= mNumFLPs; ++i) {
    stringstream out;
    out << i % 10;
    header += out.str();
    //i > 9 ? header += " " : header += "  ";
  }
  LOG(INFO) << header;

  for (auto& it : buffer) {
    string stars = "";
    for (unsigned int j = 1; j <= (it.second).parts.Size(); ++j) {
      stars += "*";
    }
    LOG(INFO) << setw(4) << it.first << ": " << stars;
  }
}

void EPNReceiver::DiscardIncompleteTimeframes()
{
  auto it = mTimeframeBuffer.begin();

  while (it != mTimeframeBuffer.end()) {
    if (duration_cast<milliseconds>(steady_clock::now() - (it->second).start).count() > mBufferTimeoutInMs) {
      LOG(WARN) << "Timeframe #" << it->first << " incomplete after " << mBufferTimeoutInMs << " milliseconds, discarding";
      mDiscardedSet.insert(it->first);
      mTimeframeBuffer.erase(it++);
      LOG(WARN) << "Number of discarded timeframes: " << mDiscardedSet.size();
    } else {
      // LOG(INFO) << "Timeframe #" << it->first << " within timeout, buffering...";
      ++it;
    }
  }
}

void EPNReceiver::Run()
{
  // DEBUG: store receive intervals per FLP
  // vector<vector<int>> rcvIntervals(fNumFLPs, vector<int>());
  // vector<std::chrono::steady_clock::time_point> rcvTimestamp(fNumFLPs);
  // end DEBUG

  // f2eHeader* header; // holds the header of the currently arrived message.
  uint16_t id = 0; // holds the timeframe id of the currently arrived sub-timeframe.

  FairMQChannel& ackOutChannel = fChannels.at(mAckChannelName).at(0);

  while (compatibility::FairMQ13<FairMQDevice>::IsRunning(this)) {
    FairMQParts parts;

    if (Receive(parts, mInChannelName, 0, 100) > 0) {
      // store the received ID
      f2eHeader& header = *(static_cast<f2eHeader*>(parts.At(0)->GetData()));
      id = header.timeFrameId;
      // LOG(INFO) << "Received sub-time frame #" << id << " from FLP" << header.flpIndex;

      // DEBUG:: store receive intervals per FLP
      // if (fTestMode > 0) {
      //   int flpId = header.flpIndex;
      //   rcvIntervals.at(flpId).push_back(duration_cast<milliseconds>(steady_clock::now() - rcvTimestamp.at(flpId)).count());
      //   LOG(WARN) << rcvIntervals.at(flpId).back();
      //   rcvTimestamp.at(flpId) = steady_clock::now();
      // }
      // end DEBUG

      if (mDiscardedSet.find(id) == mDiscardedSet.end()) {
        if (mTimeframeBuffer.find(id) == mTimeframeBuffer.end()) {
          // if this is the first part with this ID, save the receive time.
          mTimeframeBuffer[id].start = steady_clock::now();
        }
        // if the received ID has not previously been discarded,
        // store the data part in the buffer
        mTimeframeBuffer[id].parts.AddPart(move(parts.At(1)));
        // PrintBuffer(fTimeframeBuffer);
      } else {
        // if received ID has been previously discarded.
        LOG(WARN) << "Received part from an already discarded timeframe with id " << id;
      }

      if (mTimeframeBuffer[id].parts.Size() == mNumFLPs) {
        if (mTestMode > 0) {
          // Send an acknowledgement back to the sampler to measure the round trip time
          unique_ptr<FairMQMessage> ack(NewMessage(sizeof(uint16_t)));
          memcpy(ack->GetData(), &id, sizeof(uint16_t));

          if (ackOutChannel.Send(ack, 0) <= 0) {
            LOG(ERROR) << "Could not send acknowledgement without blocking";
          }
        } else {
          // LOG(INFO) << "Collected all parts for timeframe #" << id;
          // when all parts are collected send them to the output channel
          Send(mTimeframeBuffer[id].parts, mOutChannelName);
        }

        // fTimeframeBuffer[id].end = steady_clock::now();

        mTimeframeBuffer.erase(id);
      }

      // LOG(WARN) << "Buffer size: " << fTimeframeBuffer.size();
    }

    // check if any incomplete timeframes in the buffer are older than timeout period, and discard them if they are
    DiscardIncompleteTimeframes();
  }

  // DEBUG: save
  // if (fTestMode > 0) {
  //   std::time_t t = system_clock::to_time_t(system_clock::now());
  //   tm utc = *gmtime(&t);
  //   std::stringstream s;
  //   s << utc.tm_year + 1900 << "-" << utc.tm_mon + 1 << "-" << utc.tm_mday << "-" << utc.tm_hour << "-" << utc.tm_min << "-" << utc.tm_sec;
  //   string name = s.str();
  //   for (int x = 0; x < fNumFLPs; ++x) {
  //     ofstream flpRcvTimes(fId + "-" + name + "-flp-" + to_string(x) + ".log");
  //     for (auto it = rcvIntervals.at(x).begin() ; it != rcvIntervals.at(x).end(); ++it) {
  //       flpRcvTimes << *it << endl;
  //     }
  //     flpRcvTimes.close();
  //   }
  // }
  // end DEBUG
}
