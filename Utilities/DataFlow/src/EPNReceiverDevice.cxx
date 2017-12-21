// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include <cstddef> // size_t
#include <fstream> // writing to file (DEBUG)
#include <cstring>

#include <FairMQLogger.h>
#include <options/FairMQProgOptions.h>

#include "DataFlow/EPNReceiverDevice.h"
#include "Headers/DataHeader.h"
#include "Headers/SubframeMetadata.h"
#include "TimeFrame/TimeFrame.h"

#include <iomanip>

using namespace std;
using namespace std::chrono;
using namespace o2::Devices;
using SubframeMetadata = o2::DataFlow::SubframeMetadata;
using TPCTestPayload = o2::DataFlow::TPCTestPayload;
using TPCTestCluster = o2::DataFlow::TPCTestCluster;
using IndexElement = o2::DataFormat::IndexElement;

void EPNReceiverDevice::InitTask()
{
  mNumFLPs = GetConfig()->GetValue<int>("num-flps");
  mBufferTimeoutInMs = GetConfig()->GetValue<int>("buffer-timeout");
  mTestMode = GetConfig()->GetValue<int>("test-mode");
  mInChannelName = GetConfig()->GetValue<string>("in-chan-name");
  mOutChannelName = GetConfig()->GetValue<string>("out-chan-name");
  mAckChannelName = GetConfig()->GetValue<string>("ack-chan-name");
}

void EPNReceiverDevice::PrintBuffer(const unordered_map<uint16_t, TFBuffer>& buffer) const
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

void EPNReceiverDevice::DiscardIncompleteTimeframes()
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

void EPNReceiverDevice::Run()
{
  uint16_t id = 0; // holds the timeframe id of the currently arrived sub-timeframe.

  FairMQChannel& ackOutChannel = fChannels.at(mAckChannelName).at(0);

  // Simple multi timeframe index
  using TimeframeId = int;
  using FlpId = int;
  std::multimap<TimeframeId, IndexElement> index;
  std::multimap<TimeframeId, FlpId> flpIds;

  while (CheckCurrentState(RUNNING)) {
    FairMQParts subtimeframeParts;
    if (Receive(subtimeframeParts, mInChannelName, 0, 100) <= 0)
      continue;

    assert(subtimeframeParts.Size() >= 2);

    const auto* dh = o2::header::get<header::DataHeader>(subtimeframeParts.At(0)->GetData());
    assert(strncmp(dh->dataDescription.str, "SUBTIMEFRAMEMD", 16) == 0);
    SubframeMetadata* sfm = reinterpret_cast<SubframeMetadata*>(subtimeframeParts.At(1)->GetData());
    id = o2::DataFlow::timeframeIdFromTimestamp(sfm->startTime, sfm->duration);
    auto flpId = sfm->flpIndex;

    if (mDiscardedSet.find(id) == mDiscardedSet.end())
    {
      if (mTimeframeBuffer.find(id) == mTimeframeBuffer.end())
      {
        // if this is the first part with this ID, save the receive time.
        mTimeframeBuffer[id].start = steady_clock::now();
      }
      flpIds.insert(std::make_pair(id, flpId));
      LOG(INFO) << "Timeframe ID " << id << " for startTime " << sfm->startTime  << "\n";
      // If the received ID has not previously been discarded, store
      // the data part in the buffer For the moment we just concatenate
      // the subtimeframes and add an index for their description at
      // the end. Given every second part is a data header we skip
      // every two parts to populate the index. Moreover we know that
      // the SubframeMetadata is always in the second part, so we can
      // extract the flpId from there.
      for (size_t i = 0; i < subtimeframeParts.Size(); ++i)
      {
        if (i % 2 == 0)
        {
          const auto * adh = o2::header::get<header::DataHeader>(subtimeframeParts.At(i)->GetData());
          auto ie = std::make_pair(*adh, index.count(id)*2);
          index.insert(std::make_pair(id, ie));
        }
        mTimeframeBuffer[id].parts.AddPart(move(subtimeframeParts.At(i)));
      }
    }
    else
    {
      // if received ID has been previously discarded.
      LOG(WARN) << "Received part from an already discarded timeframe with id " << id;
    }

    if (flpIds.count(id) == mNumFLPs) {
      LOG(INFO) << "Timeframe " << id << " complete. Publishing.\n";
      o2::header::DataHeader tih;
      std::vector<IndexElement> flattenedIndex;

      tih.dataDescription = o2::header::DataDescription("TIMEFRAMEINDEX");
      tih.dataOrigin = o2::header::DataOrigin("EPN");
      tih.subSpecification = 0;
      tih.payloadSize = index.count(id) * sizeof(flattenedIndex.front());
      void *indexData = malloc(tih.payloadSize);
      auto indexRange = index.equal_range(id);
      for (auto ie = indexRange.first; ie != indexRange.second; ++ie)
      {
        flattenedIndex.push_back(ie->second);
      }
      memcpy(indexData, flattenedIndex.data(), tih.payloadSize);

      mTimeframeBuffer[id].parts.AddPart(NewSimpleMessage(tih));
      mTimeframeBuffer[id].parts.AddPart(NewMessage(indexData, tih.payloadSize,
                         [](void* data, void* hint){ free(data); }, nullptr));
      // LOG(INFO) << "Collected all parts for timeframe #" << id;
      // when all parts are collected send then to the output channel
      Send(mTimeframeBuffer[id].parts, mOutChannelName);
      LOG(INFO) << "Index count for " << id << " " << index.count(id) << "\n";
      index.erase(id);
      LOG(INFO) << "Index count for " << id << " " << index.count(id) << "\n";
      flpIds.erase(id);

      if (mTestMode > 0) {
        // Send an acknowledgement back to the sampler to measure the round trip time
        unique_ptr<FairMQMessage> ack(NewMessage(sizeof(uint16_t)));
        memcpy(ack->GetData(), &id, sizeof(uint16_t));

        if (ackOutChannel.Send(ack, 0) <= 0) {
          LOG(ERROR) << "Could not send acknowledgement without blocking";
        }
      }

      mTimeframeBuffer.erase(id);
    }

    // LOG(WARN) << "Buffer size: " << fTimeframeBuffer.size();

    // Check if any incomplete timeframes in the buffer are older than
    // timeout period, and discard them if they are
    // QUESTION: is this really what we want to do?
    DiscardIncompleteTimeframes();
  }
}
