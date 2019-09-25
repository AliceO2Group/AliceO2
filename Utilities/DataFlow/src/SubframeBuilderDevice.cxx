// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   SubframeBuilderDevice.cxx
/// @author Giulio Eulisse, Matthias Richter, Sandro Wenzel
/// @since  2017-02-07
/// @brief  Demonstrator device for a subframe builder

#include <thread> // this_thread::sleep_for
#include <chrono>
#include <functional>

#include "DataFlow/SubframeBuilderDevice.h"
#include "DataFlow/SubframeUtils.h"
#include "Headers/SubframeMetadata.h"
#include "Headers/HeartbeatFrame.h"
#include "Headers/DataHeader.h"
#include <options/FairMQProgOptions.h>

using HeartbeatHeader = o2::header::HeartbeatHeader;
using HeartbeatTrailer = o2::header::HeartbeatTrailer;
using DataHeader = o2::header::DataHeader;
using SubframeId = o2::dataflow::SubframeId;

o2::data_flow::SubframeBuilderDevice::SubframeBuilderDevice()
  : O2Device()
{
}

o2::data_flow::SubframeBuilderDevice::~SubframeBuilderDevice() = default;

void o2::data_flow::SubframeBuilderDevice::InitTask()
{
  mOrbitDuration = GetConfig()->GetValue<uint32_t>(OptionKeyOrbitDuration);
  mOrbitsPerTimeframe = GetConfig()->GetValue<uint32_t>(OptionKeyOrbitsPerTimeframe);
  mInputChannelName = GetConfig()->GetValue<std::string>(OptionKeyInputChannelName);
  mOutputChannelName = GetConfig()->GetValue<std::string>(OptionKeyOutputChannelName);
  mFLPId = GetConfig()->GetValue<size_t>(OptionKeyFLPId);
  mStripHBF = GetConfig()->GetValue<bool>(OptionKeyStripHBF);

  LOG(INFO) << "Obtaining data from DataPublisher\n";
  // Now that we have all the information lets create the policies to do the
  // payload extraction and merging and create the actual PayloadMerger.

  // We extract the timeframeId from the number of orbits.
  // FIXME: handle multiple socket ids
  Merger::IdExtractor makeId = [this](std::unique_ptr<FairMQMessage>& msg) -> SubframeId {
    HeartbeatHeader* hbh = reinterpret_cast<HeartbeatHeader*>(msg->GetData());
    SubframeId id = {.timeframeId = hbh->orbit / this->mOrbitsPerTimeframe,
                     .socketId = 0};
    return id;
  };

  // We extract the payload differently depending on wether we want to strip
  // the header or not.
  Merger::PayloadExtractor payloadExtractor = [this](char** out, char* in, size_t inSize) -> size_t {
    if (!this->mStripHBF) {
      return Merger::fullPayloadExtractor(out, in, inSize);
    }
    return o2::dataflow::extractDetectorPayloadStrip(out, in, inSize);
  };

  // Whether a given timeframe is complete depends on how many orbits per
  // timeframe we want.
  Merger::MergeCompletionCheker checkIfComplete =
    [this](Merger::MergeableId id, Merger::MessageMap& map) {
      return map.count(id) < this->mOrbitsPerTimeframe;
    };

  mMerger.reset(new Merger(makeId, checkIfComplete, payloadExtractor));
  OnData(mInputChannelName.c_str(), &o2::data_flow::SubframeBuilderDevice::HandleData);
}

bool o2::data_flow::SubframeBuilderDevice::BuildAndSendFrame(FairMQParts& inParts)
{
  auto id = mMerger->aggregate(inParts.At(1));

  char** outBuffer;
  size_t outSize = mMerger->finalise(outBuffer, id);
  // In this case we do not have enough subtimeframes for id,
  // so we simply return.
  if (outSize == 0)
    return true;
  // If we reach here, it means we do have enough subtimeframes.

  // top level subframe header, the DataHeader is going to be used with
  // description "SUBTIMEFRAMEMD"
  // this should be defined in a common place, and also the origin
  // the origin can probably name a detector identifier, but not sure if
  // all CRUs of a FLP in all cases serve a single detector
  o2::header::DataHeader dh;
  dh.dataDescription = o2::header::DataDescription("SUBTIMEFRAMEMD");
  dh.dataOrigin = o2::header::DataOrigin("FLP");
  dh.subSpecification = mFLPId;
  dh.payloadSize = sizeof(SubframeMetadata);

  DataHeader payloadheader(*o2::header::get<DataHeader*>((byte*)inParts.At(0)->GetData()));

  // subframe meta information as payload
  SubframeMetadata md;
  // id is really the first orbit in the timeframe.
  md.startTime = id.timeframeId * mOrbitsPerTimeframe * static_cast<uint64_t>(mOrbitDuration);
  md.duration = mOrbitDuration * mOrbitsPerTimeframe;
  LOG(INFO) << "Start time for subframe (" << md.startTime << ", "
            << md.duration
            << ")";

  // Add the metadata about the merged subtimeframes
  // FIXME: do we really need this?
  O2Message outgoing;
  o2::base::addDataBlock(outgoing, dh, NewSimpleMessage(md));

  // Add the actual merged payload.
  o2::base::addDataBlock(outgoing, payloadheader,
                         NewMessage(
                           *outBuffer, outSize,
                           [](void* data, void* hint) { delete[] reinterpret_cast<char*>(hint); }, *outBuffer));
  // send message
  Send(outgoing, mOutputChannelName.c_str());
  // FIXME: do we actually need this? outgoing should go out of scope
  outgoing.fParts.clear();

  return true;
}

bool o2::data_flow::SubframeBuilderDevice::HandleData(FairMQParts& msgParts, int /*index*/)
{
  // loop over header payload pairs in the incoming multimessage
  // for each pair
  // - check timestamp
  // - create new subtimeframe if none existing where the timestamp of the data fits
  // - add pair to the corresponding subtimeframe

  // check for completed subtimeframes and send all completed frames
  // the builder does not implement the routing to the EPN, this is done in the
  // specific FLP-EPN setup
  // to fit into the simple emulation of event/frame ids in the flpSender the order of
  // subtimeframes needs to be preserved
  BuildAndSendFrame(msgParts);
  return true;
}
