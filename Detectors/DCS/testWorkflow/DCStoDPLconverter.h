// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_DCS_TO_DPL_CONVERTER
#define O2_DCS_TO_DPL_CONVERTER

#include "Framework/DataSpecUtils.h"
#include "Framework/ExternalFairMQDeviceProxy.h"
#include <fairmq/FairMQParts.h>
#include <fairmq/FairMQDevice.h>
#include "DetectorsDCS/DataPointIdentifier.h"
#include "DetectorsDCS/DataPointValue.h"
#include "DetectorsDCS/DataPointCompositeObject.h"
#include <unordered_map>
#include <functional>
#include <string_view>

namespace o2h = o2::header;
namespace o2f = o2::framework;

// we need to provide hash function for the DataDescription
namespace std
{
template <>
struct hash<o2h::DataDescription> {
  std::size_t operator()(const o2h::DataDescription& d) const noexcept
  {
    return std::hash<std::string_view>{}({d.str, size_t(d.size)});
  }
};
} // namespace std

namespace o2
{
namespace dcs
{
using DPID = o2::dcs::DataPointIdentifier;
using DPVAL = o2::dcs::DataPointValue;
using DPCOM = o2::dcs::DataPointCompositeObject;

/// A callback function to retrieve the FairMQChannel name to be used for sending
/// messages of the specified OutputSpec

o2f::InjectorFunction dcs2dpl(std::unordered_map<DPID, o2h::DataDescription>& dpid2group, uint64_t startTime, uint64_t step, bool verbose = false)
{

  auto timesliceId = std::make_shared<size_t>(startTime);
  return [dpid2group, timesliceId, step, verbose](FairMQDevice& device, FairMQParts& parts, o2f::ChannelRetriever channelRetriever) {
    static std::unordered_map<DPID, DPCOM> cache; // will keep only the latest measurement in the 1-second wide window for each DPID
    static auto timer = std::chrono::high_resolution_clock::now();

    LOG(DEBUG) << "In lambda function: ********* Size of unordered_map (--> number of defined groups) = " << dpid2group.size();
    // We first iterate over the parts of the received message
    for (size_t i = 0; i < parts.Size(); ++i) {             // DCS sends only 1 part, but we should be able to receive more
      auto nDPCOM = parts.At(i)->GetSize() / sizeof(DPCOM); // number of DPCOM in current part
      for (size_t j = 0; j < nDPCOM; j++) {
        const auto& src = *(reinterpret_cast<const DPCOM*>(parts.At(i)->GetData()) + j);
        // do we want to check if this DP was requested ?
        auto mapEl = dpid2group.find(src.id);
        if (verbose) {
          LOG(INFO) << "Received DP " << src.id << " (data = " << src.data << "), matched to output-> " << (mapEl == dpid2group.end() ? "none " : mapEl->second.as<std::string>());
        }
        if (mapEl != dpid2group.end()) {
          auto& dst = cache[src.id] = src; // this is needed in case in the 1s window we get a new value for the same DP
        }
      }
    }

    auto timerNow = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::ratio<1>> duration = timerNow - timer;
    if (duration.count() > 1) { //did we accumulate for 1 sec?
      *timesliceId += step;     // we increment only if we send something
      std::unordered_map<o2h::DataDescription, vector<DPCOM>, std::hash<o2h::DataDescription>> outputs;
      // in the cache we have the final values of the DPs that we should put in the output
      // distribute DPs over the vectors for each requested output
      for (auto& it : cache) {
        auto mapEl = dpid2group.find(it.first);
        if (mapEl != dpid2group.end()) {
          outputs[mapEl->second].push_back(it.second);
        }
      }

      // create and send output messages
      for (auto& it : outputs) {
        o2h::DataHeader hdr(it.first, "DCS", 0);
        o2f::OutputSpec outsp{hdr.dataOrigin, hdr.dataDescription, hdr.subSpecification};
        if (it.second.empty()) {
          LOG(WARNING) << "No data for OutputSpec " << outsp;
          continue;
        }
        auto channel = channelRetriever(outsp, *timesliceId);
        if (channel.empty()) {
          LOG(WARNING) << "No output channel found for OutputSpec " << outsp << ", discarding its data";
          it.second.clear();
          continue;
        }

        hdr.tfCounter = *timesliceId; // this also
        hdr.payloadSerializationMethod = o2h::gSerializationMethodNone;
        hdr.splitPayloadParts = 1;
        hdr.splitPayloadIndex = 1;
        hdr.payloadSize = it.second.size() * sizeof(DPCOM);
        hdr.firstTForbit = 0; // this should be irrelevant for DCS
        o2h::Stack headerStack{hdr, o2::framework::DataProcessingHeader{*timesliceId, 0}};
        auto fmqFactory = device.GetChannel(channel).Transport();
        auto hdMessage = fmqFactory->CreateMessage(headerStack.size(), fair::mq::Alignment{64});
        auto plMessage = fmqFactory->CreateMessage(hdr.payloadSize, fair::mq::Alignment{64});
        memcpy(hdMessage->GetData(), headerStack.data(), headerStack.size());
        memcpy(plMessage->GetData(), it.second.data(), hdr.payloadSize);
        if (verbose) {
          LOG(INFO) << "Pushing " << it.second.size() << " DPs to output " << it.first.as<std::string>() << " for TimeSlice " << *timesliceId;
          hdr.print();
        }
        it.second.clear();
        FairMQParts outParts;
        outParts.AddPart(std::move(hdMessage));
        outParts.AddPart(std::move(plMessage));
        o2f::sendOnChannel(device, outParts, channel);
      }

      timer = timerNow;
      cache.clear();
    }
  };
}

} // namespace dcs
} // namespace o2

#endif /* O2_DCS_TO_DPL_CONVERTER_H */
