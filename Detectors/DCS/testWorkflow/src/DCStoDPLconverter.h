// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_DCS_TO_DPL_CONVERTER
#define O2_DCS_TO_DPL_CONVERTER

#include "Framework/DataSpecUtils.h"
#include "Framework/ExternalFairMQDeviceProxy.h"
#include <fairmq/Parts.h>
#include <fairmq/Device.h>
#include "DetectorsDCS/DataPointIdentifier.h"
#include "DetectorsDCS/DataPointValue.h"
#include "DetectorsDCS/DataPointCompositeObject.h"
#include "CommonUtils/StringUtils.h"
#include <unordered_map>
#include <functional>
#include <string_view>
#include <chrono>

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

o2f::InjectorFunction dcs2dpl(std::unordered_map<DPID, o2h::DataDescription>& dpid2group, bool fbiFirst, bool verbose = false)
{

  return [dpid2group, fbiFirst, verbose](o2::framework::TimingInfo& tinfo, fair::mq::Device& device, fair::mq::Parts& parts, o2f::ChannelRetriever channelRetriever) {
    static std::unordered_map<DPID, DPCOM> cache; // will keep only the latest measurement in the 1-second wide window for each DPID
    static auto timer = std::chrono::high_resolution_clock::now();
    static auto timer0 = std::chrono::high_resolution_clock::now();
    static bool seenFBI = false;
    static uint32_t localTFCounter = 0;
    static size_t nInp = 0, nInpFBI = 0;
    static size_t szInp = 0, szInpFBI = 0;
    LOG(debug) << "In lambda function: ********* Size of unordered_map (--> number of defined groups) = " << dpid2group.size();
    // check if we got FBI (Master) or delta (MasterDelta)
    if (!parts.Size()) {
      LOGP(warn, "Empty input recieved at timeslice {}", tinfo.timeslice);
      return;
    }
    std::string firstName = std::string((char*)&(reinterpret_cast<const DPCOM*>(parts.At(0)->GetData()))->id);

    bool isFBI = false;
    nInp++;
    if (o2::utils::Str::endsWith(firstName, "Master")) {
      isFBI = true;
      nInpFBI++;
      seenFBI = true;
    } else if (o2::utils::Str::endsWith(firstName, "MasterDelta")) {
      isFBI = false;
    } else {
      LOGP(error, "Cannot determine if the map is FBI or Delta, 1st DP name is {}", firstName);
    }
    if (verbose) {
      LOGP(info, "New input of {} parts received, map type: {}, timeslice {}", parts.Size(), isFBI ? "FBI" : "Delta", tinfo.timeslice);
    }

    // We first iterate over the parts of the received message
    for (size_t i = 0; i < parts.Size(); ++i) {             // DCS sends only 1 part, but we should be able to receive more
      auto sz = parts.At(i)->GetSize();
      szInp += sz;
      if (isFBI) {
        szInpFBI += sz;
      }
      auto nDPCOM = sz / sizeof(DPCOM); // number of DPCOM in current part
      for (size_t j = 0; j < nDPCOM; j++) {
        const auto& src = *(reinterpret_cast<const DPCOM*>(parts.At(i)->GetData()) + j);
        // do we want to check if this DP was requested ?
        auto mapEl = dpid2group.find(src.id);
        if (verbose) {
          LOG(info) << "Received DP " << src.id << " (data = " << src.data << "), matched to output-> " << (mapEl == dpid2group.end() ? "none " : mapEl->second.as<std::string>());
        }
        if (mapEl != dpid2group.end()) {
          cache[src.id] = src; // this is needed in case in the 1s window we get a new value for the same DP
        }
      }
    }
    auto timerNow = std::chrono::high_resolution_clock::now();
    if (fbiFirst && nInpFBI < 2) { // 1st FBI might be obsolete
      seenFBI = false;
      static int prevDelay = 0;
      std::chrono::duration<double, std::ratio<1>> duration = timerNow - timer0;
      int delay = duration.count();
      if (delay > prevDelay) {
        LOGP(info, "Waiting for requested 1st FBI since {} s", delay);
        prevDelay = delay;
      }
    }

    std::chrono::duration<double, std::ratio<1>> duration = timerNow - timer;
    if (duration.count() > 1 && (seenFBI || !fbiFirst)) { // did we accumulate for 1 sec and have we seen FBI if it was requested?
      std::unordered_map<o2h::DataDescription, pmr::vector<DPCOM>, std::hash<o2h::DataDescription>> outputs;
      // in the cache we have the final values of the DPs that we should put in the output
      // distribute DPs over the vectors for each requested output
      for (auto& it : cache) {
        auto mapEl = dpid2group.find(it.first);
        if (mapEl != dpid2group.end()) {
          outputs[mapEl->second].push_back(it.second);
        }
      }
      std::uint64_t creation = std::chrono::time_point_cast<std::chrono::milliseconds>(timerNow).time_since_epoch().count();
      std::unordered_map<std::string, std::unique_ptr<fair::mq::Parts>> messagesPerRoute;
      // create and send output messages
      for (auto& it : outputs) { // distribute messages per routes
        o2h::DataHeader hdr(it.first, "DCS", 0);
        o2f::OutputSpec outsp{hdr.dataOrigin, hdr.dataDescription, hdr.subSpecification};
        if (it.second.empty()) {
          LOG(warning) << "No data for OutputSpec " << outsp;
          continue;
        }
        auto channel = channelRetriever(outsp, tinfo.timeslice);
        if (channel.empty()) {
          LOG(warning) << "No output channel found for OutputSpec " << outsp << ", discarding its data";
          it.second.clear();
          continue;
        }

        hdr.tfCounter = localTFCounter; // this also
        hdr.payloadSerializationMethod = o2h::gSerializationMethodNone;
        hdr.splitPayloadParts = 1;
        hdr.splitPayloadIndex = 1;
        hdr.payloadSize = it.second.size() * sizeof(DPCOM);
        hdr.firstTForbit = 0; // this should be irrelevant for DCS
        o2h::Stack headerStack{hdr, o2::framework::DataProcessingHeader{tinfo.timeslice, 1, creation}};
        auto fmqFactory = device.GetChannel(channel).Transport();
        auto hdMessage = fmqFactory->CreateMessage(headerStack.size(), fair::mq::Alignment{64});
        auto plMessage = fmqFactory->CreateMessage(hdr.payloadSize, fair::mq::Alignment{64});
        memcpy(hdMessage->GetData(), headerStack.data(), headerStack.size());
        memcpy(plMessage->GetData(), it.second.data(), hdr.payloadSize);

        fair::mq::Parts* parts2send = messagesPerRoute[channel].get(); // fair::mq::Parts*
        if (!parts2send) {
          messagesPerRoute[channel] = std::make_unique<fair::mq::Parts>();
          parts2send = messagesPerRoute[channel].get();
        }
        parts2send->AddPart(std::move(hdMessage));
        parts2send->AddPart(std::move(plMessage));
        if (verbose) {
          LOGP(info, "Pushing {} DPs to {} for TimeSlice {} at {}", it.second.size(), o2f::DataSpecUtils::describe(outsp), tinfo.timeslice, creation);
        }
        it.second.clear();
      }
      // push output of every route
      for (auto& msgIt : messagesPerRoute) {
        LOG(info) << "Sending " << msgIt.second->Size() / 2 << " parts to channel " << msgIt.first;
        o2f::sendOnChannel(device, *msgIt.second.get(), msgIt.first, tinfo.timeslice);
      }
      timer = timerNow;
      cache.clear();
      if (!messagesPerRoute.empty()) {
        localTFCounter++;
      }
    }
    if (isFBI) {
      float runtime = 1e-3 * std::chrono::duration_cast<std::chrono::milliseconds>(timerNow - timer0).count();
      LOGP(info, "{} inputs ({} bytes) of which {} FBI ({} bytes) seen in {:.3f} s", nInp, fmt::group_digits(szInp), nInpFBI, fmt::group_digits(szInpFBI), runtime);
    }
  };
}

} // namespace dcs
} // namespace o2

#endif /* O2_DCS_TO_DPL_CONVERTER_H */
