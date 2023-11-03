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
#include "Framework/AlgorithmSpec.h"
#include "Framework/ConfigParamSpec.h"
#include "Framework/AlgorithmSpec.h"
#include "Framework/DataProcessingHeader.h"
#include "Framework/DataSpecUtils.h"
#include "Framework/DeviceSpec.h"
#include "Framework/ExternalFairMQDeviceProxy.h"
#include "Framework/InitContext.h"
#include "Framework/ProcessingContext.h"
#include "Framework/RawDeviceService.h"
#include "Framework/DeviceContext.h"
#include "Framework/CallbackService.h"
#include "Framework/ControlService.h"
#include "Framework/SourceInfoHeader.h"
#include "Framework/ChannelInfo.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/RateLimiter.h"
#include "Framework/TimesliceIndex.h"
#include "Framework/TimingInfo.h"
#include "Framework/DeviceState.h"
#include "Framework/Monitoring.h"
#include "Headers/DataHeader.h"
#include "Headers/Stack.h"
#include "DecongestionService.h"
#include "CommonConstants/LHCConstants.h"

#include "./DeviceSpecHelpers.h"
#include "Monitoring/Monitoring.h"

#include <fairmq/Parts.h>
#include <fairmq/Device.h>
#include <uv.h>
#include <cstring>
#include <cassert>
#include <memory>
#include <optional>
#include <unordered_map>
#include <numeric> // std::accumulate
#include <sstream>
#include <stdexcept>
#include <regex>

namespace o2::framework
{
using DataHeader = o2::header::DataHeader;

std::string formatExternalChannelConfiguration(InputChannelSpec const& spec)
{
  return DeviceSpecHelpers::inputChannel2String(spec);
}

std::string formatExternalChannelConfiguration(OutputChannelSpec const& spec)
{
  return DeviceSpecHelpers::outputChannel2String(spec);
}

std::string formatExternalChannelConfiguration(OutputChannelSpec const&);

void sendOnChannel(fair::mq::Device& device, fair::mq::Parts& messages, std::string const& channel, size_t timeSlice)
{
  // Note: DPL is only setting up one instance of a channel while FairMQ allows to have an
  // array of channels, the index is 0 in the call
  constexpr auto index = 0;
  LOG(debug) << "sending " << messages.Size() << " messages on " << channel;
  // TODO: we can make this configurable
  const int maxTimeout = 10000;
  int timeout = 0;
  // try dispatch with increasing timeout in order to also drop a warning if the dispatching
  // has been tried multiple times within max timeout
  // since we do not want any messages to be dropped at this stage, we stay in the loop until
  // the downstream congestion is resolved
  // TODO: we might want to treat this error condition some levels higher up, but for
  // the moment its an appropriate solution. The important thing is not to drop
  // messages and to be informed about the congestion.
  while (device.Send(messages, channel, index, timeout) < 0) {
    if (timeout == 0) {
      timeout = 1;
    } else if (timeout < maxTimeout) {
      timeout *= 10;
    } else {
      LOG(alarm) << "Cannot dispatch to channel " << channel << " due to DOWNSTREAM BACKPRESSURE. NO DATA IS DROPPED,"
                 << " will keep retrying. This is only a problem if downstream congestion does not resolve by itself.";
      if (timeout == maxTimeout) {
        // we add 1ms to disable the warning below
        timeout += 1;
      }
    }
    if (device.NewStatePending()) {
      LOG(alarm) << "Device state change is requested, dropping " << messages.Size() << " pending message(s) "
                 << "on channel " << channel << ". "
                 << "ATTENTION: DATA IS LOST! Could not dispatch data to downstream consumer(s), check if "
                 << "consumers have been terminated too early";
      // make sure we disable the warning below
      timeout = maxTimeout + 1;
      break;
    }
  }

  // FIXME: we need a better logic for avoiding message spam
  if (timeout > 1 && timeout <= maxTimeout) {
    LOG(warning) << "dispatching on channel " << channel << " was delayed by " << timeout << " ms";
  }
  // TODO: feeling this is a bit awkward, but the interface of fair::mq::Parts does not provide a
  // method to clear the content.
  // Maybe the FairMQ API can be improved at some point. Actually the ownership of all messages should be passed
  // on to the transport and the messages should be empty after sending and the parts content can be cleared.
  // assert(std::accumulate(messages.begin(), messages.end(), true, [](bool a, auto const& msg) {return a && (msg.get() == nullptr);}));
  messages.fParts.clear();
}

void sendOnChannel(fair::mq::Device& device, fair::mq::Parts& messages, OutputSpec const& spec, DataProcessingHeader::StartTime tslice, ChannelRetriever& channelRetriever)
{
  // Note: DPL is only setting up one instance of a channel while FairMQ allows to have an
  // array of channels, the index is 0 in the call
  auto channel = channelRetriever(spec, tslice);
  if (channel.empty()) {
    LOG(warning) << "can not find matching channel for " << DataSpecUtils::describe(spec) << " timeslice " << tslice;
    return;
  }
  sendOnChannel(device, messages, channel, tslice);
}

void sendOnChannel(fair::mq::Device& device, o2::header::Stack&& headerStack, fair::mq::MessagePtr&& payloadMessage, OutputSpec const& spec, ChannelRetriever& channelRetriever)
{
  const auto* dph = o2::header::get<DataProcessingHeader*>(headerStack.data());
  if (!dph) {
    LOG(error) << "Header Stack does not follow the O2 data model, DataProcessingHeader missing";
    return;
  }
  auto channelName = channelRetriever(spec, dph->startTime);
  constexpr auto index = 0;
  if (channelName.empty()) {
    LOG(warning) << "can not find matching channel for " << DataSpecUtils::describe(spec);
    return;
  }
  for (auto& channelInfo : device.GetChannels()) {
    if (channelInfo.first != channelName) {
      continue;
    }
    assert(channelInfo.second.size() == 1);
    // allocate the header message using the underlying transport of the channel
    auto channelAlloc = o2::pmr::getTransportAllocator(channelInfo.second[index].Transport());
    fair::mq::MessagePtr headerMessage = o2::pmr::getMessage(std::move(headerStack), channelAlloc);

    fair::mq::Parts out;
    out.AddPart(std::move(headerMessage));
    out.AddPart(std::move(payloadMessage));
    sendOnChannel(device, out, channelName, dph->startTime);
    return;
  }
  LOG(error) << "internal mismatch, can not find channel " << channelName << " in the list of channel infos of the device";
}

void sendOnChannel(fair::mq::Device& device, fair::mq::MessagePtr&& headerMessage, fair::mq::MessagePtr&& payloadMessage, OutputSpec const& spec, ChannelRetriever& channelRetriever)
{
  //  const auto* dph = o2::header::get<DataProcessingHeader*>( *reinterpret_cast<o2::header::Stack*>(headerMessage->GetData()) );
  const auto* dph = o2::header::get<DataProcessingHeader*>(headerMessage->GetData());
  if (!dph) {
    LOG(error) << "Header does not follow the O2 data model, DataProcessingHeader missing";
    return;
  }
  auto tslice = dph->startTime;
  fair::mq::Parts out;
  out.AddPart(std::move(headerMessage));
  out.AddPart(std::move(payloadMessage));
  sendOnChannel(device, out, spec, tslice, channelRetriever);
}

void appendForSending(fair::mq::Device& device, o2::header::Stack&& headerStack, size_t timeSliceID, fair::mq::MessagePtr&& payloadMessage, OutputSpec const& spec, fair::mq::Parts& messageCache, ChannelRetriever& channelRetriever)
{
  auto channelName = channelRetriever(spec, timeSliceID);
  constexpr auto index = 0;
  if (channelName.empty()) {
    LOG(warning) << "can not find matching channel for " << DataSpecUtils::describe(spec);
    return;
  }
  for (auto& channelInfo : device.GetChannels()) {
    if (channelInfo.first != channelName) {
      continue;
    }
    assert(channelInfo.second.size() == 1);
    // allocate the header message using the underlying transport of the channel
    auto channelAlloc = o2::pmr::getTransportAllocator(channelInfo.second[index].Transport());
    fair::mq::MessagePtr headerMessage = o2::pmr::getMessage(std::move(headerStack), channelAlloc);

    fair::mq::Parts out;
    messageCache.AddPart(std::move(headerMessage));
    messageCache.AddPart(std::move(payloadMessage));
    return;
  }
  LOG(error) << "internal mismatch, can not find channel " << channelName << " in the list of channel infos of the device";
}

InjectorFunction o2DataModelAdaptor(OutputSpec const& spec, uint64_t startTime, uint64_t /*step*/)
{
  return [spec](TimingInfo&, ServiceRegistryRef const& ref, fair::mq::Parts& parts, ChannelRetriever channelRetriever, size_t newTimesliceId, bool& stop) -> bool {
    auto* device = ref.get<RawDeviceService>().device();
    for (int i = 0; i < parts.Size() / 2; ++i) {
      auto dh = o2::header::get<DataHeader*>(parts.At(i * 2)->GetData());

      DataProcessingHeader dph{newTimesliceId, 0};
      o2::header::Stack headerStack{*dh, dph};
      sendOnChannel(*device, std::move(headerStack), std::move(parts.At(i * 2 + 1)), spec, channelRetriever);
    }
    return parts.Size() > 0;
  };
}

auto getFinalIndex(DataHeader const& dh, size_t msgidx) -> size_t
{
  size_t finalBlockIndex = 0;
  if (dh.splitPayloadParts > 0 && dh.splitPayloadParts == dh.splitPayloadIndex) {
    // this is indicating a sequence of payloads following the header
    // FIXME: we will probably also set the DataHeader version
    // Current position + number of parts + 1 (for the header)
    finalBlockIndex = msgidx + dh.splitPayloadParts + 1;
  } else {
    // We can consider the next splitPayloadParts as one block of messages pairs
    // because we are guaranteed they are all the same.
    // If splitPayloadParts = 0, we assume that means there is only one (header, payload)
    // pair.
    finalBlockIndex = msgidx + (dh.splitPayloadParts > 0 ? dh.splitPayloadParts : 1) * 2;
  }
  assert(finalBlockIndex >= msgidx + 2);
  return finalBlockIndex;
};

void injectMissingData(fair::mq::Device& device, fair::mq::Parts& parts, std::vector<OutputRoute> const& routes, bool doInjectMissingData, unsigned int doPrintSizes)
{
  // Check for missing data.
  static std::vector<bool> present;
  static std::vector<size_t> dataSizes;
  static std::vector<bool> showSize;
  present.clear();
  present.resize(routes.size(), false);
  dataSizes.clear();
  dataSizes.resize(routes.size(), 0);
  showSize.clear();
  showSize.resize(routes.size(), false);

  static std::vector<size_t> unmatchedDescriptions;
  unmatchedDescriptions.clear();
  DataProcessingHeader const* dph = nullptr;
  DataHeader const* firstDH = nullptr;
  bool hassih = false;

  // Do not check anything which has DISTSUBTIMEFRAME in it.
  size_t expectedDataSpecs = 0;
  for (size_t pi = 0; pi < present.size(); ++pi) {
    auto& spec = routes[pi].matcher;
    if (DataSpecUtils::asConcreteDataTypeMatcher(spec).description == header::DataDescription("DISTSUBTIMEFRAME")) {
      present[pi] = true;
      continue;
    }
    if (routes[pi].timeslice == 0) {
      ++expectedDataSpecs;
    }
  }

  size_t foundDataSpecs = 0;
  for (int msgidx = 0; msgidx < parts.Size(); msgidx += 2) {
    bool allFound = true;
    int addToSize = -1;
    const auto dh = o2::header::get<DataHeader*>(parts.At(msgidx)->GetData());
    auto const sih = o2::header::get<SourceInfoHeader*>(parts.At(msgidx)->GetData());
    if (sih != nullptr) {
      hassih = true;
      continue;
    }
    if (parts.At(msgidx).get() == nullptr) {
      LOG(error) << "unexpected nullptr found. Skipping message pair.";
      continue;
    }
    if (!dh) {
      LOG(error) << "data on input " << msgidx << " does not follow the O2 data model, DataHeader missing";
      if (msgidx > 0) {
        --msgidx;
      }
      continue;
    }
    if (firstDH == nullptr) {
      firstDH = dh;
      if (doPrintSizes && firstDH->tfCounter % doPrintSizes != 0) {
        doPrintSizes = 0;
      }
    }
    // Copy the DataProcessingHeader from the first message.
    if (dph == nullptr) {
      dph = o2::header::get<DataProcessingHeader*>(parts.At(msgidx)->GetData());
      for (size_t pi = 0; pi < present.size(); ++pi) {
        if (routes[pi].timeslice != (dph->startTime % routes[pi].maxTimeslices)) {
          present[pi] = true;
        }
      }
    }
    for (size_t pi = 0; pi < present.size(); ++pi) {
      if (present[pi] && !doPrintSizes) {
        continue;
      }
      // Consider uninvolved pipelines as present.
      if (routes[pi].timeslice != (dph->startTime % routes[pi].maxTimeslices)) {
        present[pi] = true;
        continue;
      }
      allFound = false;
      auto& spec = routes[pi].matcher;
      OutputSpec query{dh->dataOrigin, dh->dataDescription, dh->subSpecification};
      if (DataSpecUtils::match(spec, query)) {
        if (!present[pi]) {
          ++foundDataSpecs;
          present[pi] = true;
          showSize[pi] = true;
        }
        addToSize = pi;
        break;
      }
    }
    int msgidxLast = getFinalIndex(*dh, msgidx);
    if (addToSize >= 0) {
      int increment = (dh->splitPayloadParts > 0 && dh->splitPayloadParts == dh->splitPayloadIndex) ? 1 : 2;
      for (int msgidx2 = msgidx + 1; msgidx2 < msgidxLast; msgidx2 += increment) {
        dataSizes[addToSize] += parts.At(msgidx2)->GetSize();
      }
    }
    // Skip the rest of the block of messages. We subtract 2 because above we increment by 2.
    msgidx = msgidxLast - 2;
    if (allFound && !doPrintSizes) {
      return;
    }
  }

  for (size_t pi = 0; pi < present.size(); ++pi) {
    if (!present[pi]) {
      showSize[pi] = true;
      unmatchedDescriptions.push_back(pi);
    }
  }

  if (firstDH && doPrintSizes) {
    std::string sizes = "";
    size_t totalSize = 0;
    for (size_t pi = 0; pi < present.size(); ++pi) {
      if (showSize[pi]) {
        totalSize += dataSizes[pi];
        auto& spec = routes[pi].matcher;
        sizes += DataSpecUtils::describe(spec) + fmt::format(":{} ", fmt::group_digits(dataSizes[pi]));
      }
    }
    LOGP(important, "RAW {} size report:{}- Total:{}", firstDH->tfCounter, sizes, fmt::group_digits(totalSize));
  }

  if (!doInjectMissingData) {
    return;
  }

  if (unmatchedDescriptions.size() > 0) {
    if (hassih) {
      if (firstDH) {
        LOG(error) << "Received an EndOfStream message together with data. This should not happen.";
      }
      LOG(detail) << "This is an End Of Stream message. Not injecting anything.";
      return;
    }
    if (firstDH == nullptr) {
      LOG(error) << "Input proxy received incomplete data without any data header. This should not happen! Cannot inject missing data as requsted.";
      return;
    }
    if (dph == nullptr) {
      LOG(error) << "Input proxy received incomplete data without any data processing header. This should happen! Cannot inject missing data as requsted.";
      return;
    }
    std::string missing = "";
    for (auto mi : unmatchedDescriptions) {
      auto& spec = routes[mi].matcher;
      missing += " " + DataSpecUtils::describe(spec);
      // If we have a ConcreteDataMatcher, we can create a message with the correct header.
      // If we have a ConcreteDataTypeMatcher, we use 0xdeadbeef as subSpecification.
      ConcreteDataTypeMatcher concrete = DataSpecUtils::asConcreteDataTypeMatcher(spec);
      auto subSpec = DataSpecUtils::getOptionalSubSpec(spec);
      if (subSpec == std::nullopt) {
        *subSpec = 0xDEADBEEF;
      }
      o2::header::DataHeader dh{*firstDH};
      dh.dataOrigin = concrete.origin;
      dh.dataDescription = concrete.description;
      dh.subSpecification = *subSpec;
      dh.payloadSize = 0;
      dh.splitPayloadParts = 0;
      dh.splitPayloadIndex = 0;
      dh.payloadSerializationMethod = header::gSerializationMethodNone;

      auto& channelName = routes[mi].channel;
      auto& channelInfo = device.GetChannel(channelName);
      auto channelAlloc = o2::pmr::getTransportAllocator(channelInfo.Transport());
      auto headerMessage = o2::pmr::getMessage(o2::header::Stack{channelAlloc, dh, *dph});
      parts.AddPart(std::move(headerMessage));
      // add empty payload message
      parts.AddPart(device.NewMessageFor(channelName, 0, 0));
    }
    static int maxWarn = 10; // Correct would be o2::conf::VerbosityConfig::Instance().maxWarnDeadBeef, but Framework does not depend on CommonUtils..., but not so critical since receives will send correct number of DEADBEEF messages
    static int contDeadBeef = 0;
    if (++contDeadBeef <= maxWarn) {
      LOGP(alarm, "Found {}/{} data specs, missing data specs: {}, injecting 0xDEADBEEF{}", foundDataSpecs, expectedDataSpecs, missing, contDeadBeef == maxWarn ? " - disabling alarm now to stop flooding the log" : "");
    }
  }
}

InjectorFunction dplModelAdaptor(std::vector<OutputSpec> const& filterSpecs, DPLModelAdapterConfig config)
{
  bool throwOnUnmatchedInputs = config.throwOnUnmatchedInputs;
  // structure to hold information on the unmatched data and print a warning at cleanup
  class DroppedDataSpecs
  {
   public:
    DroppedDataSpecs() = default;
    ~DroppedDataSpecs()
    {
      warning();
    }

    [[nodiscard]] bool find(std::string const& desc) const
    {
      return descriptions.find(desc) != std::string::npos;
    }

    void add(std::string const& desc)
    {
      descriptions += "\n   " + desc;
    }

    void warning() const
    {
      if (not descriptions.empty()) {
        LOG(warning) << "Some input data could not be matched by filter rules to output specs\n"
                     << "Active rules: " << descriptions << "\n"
                     << "DROPPING OF THESE MESSAGES HAS BEEN ENABLED BY CONFIGURATION";
      }
    }

   private:
    std::string descriptions;
  };

  return [filterSpecs = std::move(filterSpecs), throwOnUnmatchedInputs, droppedDataSpecs = std::make_shared<DroppedDataSpecs>()](TimingInfo& timingInfo, ServiceRegistryRef const& services, fair::mq::Parts& parts, ChannelRetriever channelRetriever, size_t newTimesliceId, bool& stop) {
    // FIXME: this in not thread safe, but better than an alloc of a map per message...
    std::unordered_map<std::string, fair::mq::Parts> outputs;
    std::vector<std::string> unmatchedDescriptions;
    auto* device = services.get<RawDeviceService>().device();

    static bool override_creation_env = getenv("DPL_RAWPROXY_OVERRIDE_ORBITRESET");
    bool override_creation = false;
    uint64_t creationVal = 0;
    if (override_creation_env) {
      static uint64_t creationValBase = std::stoul(getenv("DPL_RAWPROXY_OVERRIDE_ORBITRESET"));
      creationVal = creationValBase;
      override_creation = true;
    } else {
      auto orbitResetTimeUrl = device->fConfig->GetProperty<std::string>("orbit-reset-time", "ccdb://CTP/Calib/OrbitResetTime");
      char* err = nullptr;
      creationVal = std::strtoll(orbitResetTimeUrl.c_str(), &err, 10);
      if (err && *err == 0 && creationVal) {
        override_creation = true;
      }
    }

    for (int msgidx = 0; msgidx < parts.Size(); msgidx += 2) {
      if (parts.At(msgidx).get() == nullptr) {
        LOG(error) << "unexpected nullptr found. Skipping message pair.";
        continue;
      }
      const auto dh = o2::header::get<DataHeader*>(parts.At(msgidx)->GetData());
      if (!dh) {
        LOG(error) << "data on input " << msgidx << " does not follow the O2 data model, DataHeader missing";
        if (msgidx > 0) {
          --msgidx;
        }
        continue;
      }
      auto dph = o2::header::get<DataProcessingHeader*>(parts.At(msgidx)->GetData());
      if (!dph) {
        LOG(error) << "data on input " << msgidx << " does not follow the O2 data model, DataProcessingHeader missing";
        continue;
      }
      const_cast<DataProcessingHeader*>(dph)->startTime = newTimesliceId;
      if (override_creation) {
        const_cast<DataProcessingHeader*>(dph)->creation = creationVal + (dh->firstTForbit * o2::constants::lhc::LHCOrbitNS * 0.000001f);
      }
      timingInfo.timeslice = dph->startTime;
      timingInfo.creation = dph->creation;
      timingInfo.firstTForbit = dh->firstTForbit;
      timingInfo.runNumber = dh->runNumber;
      timingInfo.tfCounter = dh->tfCounter;
      LOG(debug) << msgidx << ": " << DataSpecUtils::describe(OutputSpec{dh->dataOrigin, dh->dataDescription, dh->subSpecification}) << " part " << dh->splitPayloadIndex << " of " << dh->splitPayloadParts << "  payload " << parts.At(msgidx + 1)->GetSize();

      OutputSpec query{dh->dataOrigin, dh->dataDescription, dh->subSpecification};
      LOG(debug) << "processing " << DataSpecUtils::describe(OutputSpec{dh->dataOrigin, dh->dataDescription, dh->subSpecification}) << " time slice " << dph->startTime << " part " << dh->splitPayloadIndex << " of " << dh->splitPayloadParts;
      int finalBlockIndex = 0;
      std::string channelName = "";

      for (auto const& spec : filterSpecs) {
        // filter on the specified OutputSpecs, the default value is a ConcreteDataTypeMatcher with origin and description 'any'
        if (DataSpecUtils::match(spec, OutputSpec{{header::gDataOriginAny, header::gDataDescriptionAny}}) ||
            DataSpecUtils::match(spec, query)) {
          channelName = channelRetriever(query, dph->startTime);
          // We do not complain about DPL/EOS/0, since it's normal not to forward it.
          if (channelName.empty() && DataSpecUtils::describe(query) != "DPL/EOS/0") {
            LOG(warning) << "can not find matching channel, not able to adopt " << DataSpecUtils::describe(query);
          }
          break;
        }
      }
      finalBlockIndex = getFinalIndex(*dh, msgidx);
      if (finalBlockIndex > parts.Size()) {
        // TODO error handling
        // LOGP(error, "DataHeader::splitPayloadParts invalid");
        continue;
      }

      if (!channelName.empty()) {
        // the checks for consistency of split payload parts are of informative nature
        // forwarding happens independently
        // if (dh->splitPayloadParts > 1 && dh->splitPayloadParts != std::numeric_limits<decltype(dh->splitPayloadParts)>::max()) {
        //  if (lastSplitPartIndex == -1 && dh->splitPayloadIndex != 0) {
        //    LOG(warning) << "wrong split part index, expecting the first of " << dh->splitPayloadParts << " part(s)";
        //  } else if (dh->splitPayloadIndex != lastSplitPartIndex + 1) {
        //    LOG(warning) << "unordered split parts, expecting part " << lastSplitPartIndex + 1 << ", got " << dh->splitPayloadIndex
        //                 << " of " << dh->splitPayloadParts;
        //  } else if (channelNameForSplitParts.empty() == false && channelName != channelNameForSplitParts) {
        //    LOG(error) << "inconsistent channel for split part " << dh->splitPayloadIndex
        //               << ", matching " << channelName << ", expecting " << channelNameForSplitParts;
        //  }
        //}
        LOGP(debug, "associating {} part(s) at index {} to channel {} ({})", finalBlockIndex - msgidx, msgidx, channelName, outputs[channelName].Size());
        for (; msgidx < finalBlockIndex; ++msgidx) {
          outputs[channelName].AddPart(std::move(parts.At(msgidx)));
        }
        msgidx -= 2;
      } else {
        msgidx = finalBlockIndex - 2;
      }
      if (finalBlockIndex == 0 && !DataSpecUtils::match(query, "DPL", "EOS", 0)) {
        unmatchedDescriptions.emplace_back(DataSpecUtils::describe(query));
      }
    } // end of loop over parts

    bool didSendParts = false;
    for (auto& [channelName, channelParts] : outputs) {
      if (channelParts.Size() == 0) {
        continue;
      }
      didSendParts = true;
      sendOnChannel(*device, channelParts, channelName, newTimesliceId);
    }
    if (not unmatchedDescriptions.empty()) {
      if (throwOnUnmatchedInputs) {
        std::string descriptions;
        for (auto const& desc : unmatchedDescriptions) {
          descriptions += "\n   " + desc;
        }
        throw std::runtime_error("No matching filter rule for input data " + descriptions +
                                 "\n Add appropriate matcher(s) to dataspec definition or allow to drop unmatched data");
      } else {
        bool changed = false;
        for (auto const& desc : unmatchedDescriptions) {
          if (not droppedDataSpecs->find(desc)) {
            // a new description
            droppedDataSpecs->add(desc);
            changed = true;
          }
        }
        if (changed) {
          droppedDataSpecs->warning();
        }
      }
    }
    return didSendParts;
  };
}

InjectorFunction incrementalConverter(OutputSpec const& spec, o2::header::SerializationMethod method, uint64_t startTime, uint64_t step)
{
  auto timesliceId = std::make_shared<size_t>(startTime);
  return [timesliceId, spec, step, method](TimingInfo&, ServiceRegistryRef const& services, fair::mq::Parts& parts, ChannelRetriever channelRetriever, size_t newTimesliceId, bool&) {
    auto* device = services.get<RawDeviceService>().device();
    // We iterate on all the parts and we send them two by two,
    // adding the appropriate O2 header.
    for (int i = 0; i < parts.Size(); ++i) {
      DataHeader dh;
      dh.payloadSerializationMethod = method;

      // FIXME: this only supports fully specified output specs...
      ConcreteDataMatcher matcher = DataSpecUtils::asConcreteDataMatcher(spec);
      dh.dataOrigin = matcher.origin;
      dh.dataDescription = matcher.description;
      dh.subSpecification = matcher.subSpec;
      dh.payloadSize = parts.At(i)->GetSize();

      DataProcessingHeader dph{newTimesliceId, 0};
      if (*timesliceId != newTimesliceId) {
        LOG(fatal) << "Time slice ID provided from oldestPossible mechanism " << newTimesliceId << " is out of sync with expected value " << *timesliceId;
      }
      *timesliceId += step;
      // we have to move the incoming data
      o2::header::Stack headerStack{dh, dph};

      sendOnChannel(*device, std::move(headerStack), std::move(parts.At(i)), spec, channelRetriever);
    }
    return parts.Size();
  };
}

DataProcessorSpec specifyExternalFairMQDeviceProxy(char const* name,
                                                   std::vector<OutputSpec> const& outputs,
                                                   char const* defaultChannelConfig,
                                                   InjectorFunction converter,
                                                   uint64_t minSHM,
                                                   bool sendTFcounter,
                                                   bool doInjectMissingData,
                                                   unsigned int doPrintSizes)
{
  DataProcessorSpec spec;
  spec.name = strdup(name);
  spec.inputs = {};
  spec.outputs = outputs;
  static std::vector<std::string> channels;
  static std::vector<int> numberOfEoS(channels.size(), 0);
  static std::vector<int> eosPeersCount(channels.size(), 0);
  // The Init method will register a new "Out of band" channel and
  // attach an OnData to it which is responsible for converting incoming
  // messages into DPL messages.
  spec.algorithm = AlgorithmSpec{[converter, minSHM, deviceName = spec.name, sendTFcounter, doInjectMissingData, doPrintSizes](InitContext& ctx) {
    auto* device = ctx.services().get<RawDeviceService>().device();
    // make a copy of the output routes and pass to the lambda by move
    auto outputRoutes = ctx.services().get<RawDeviceService>().spec().outputs;
    auto outputChannels = ctx.services().get<RawDeviceService>().spec().outputChannels;
    assert(device);

    // check that the name used for registering the OnData callback corresponds
    // to the configured output channel, unfortunately we can not automatically
    // deduce this from list of channels without knowing the name, because there
    // will be multiple channels. At least we throw a more informative exception.
    // fair::mq::Device calls the custom init before the channels have been configured
    // so we do the check before starting in a dedicated callback
    auto channelConfigurationChecker = [device, deviceName, services = ctx.services()]() {
      auto& deviceState = services.get<DeviceState>();
      channels.clear();
      numberOfEoS.clear();
      eosPeersCount.clear();
      for (auto& [channelName, _] : services.get<RawDeviceService>().device()->GetChannels()) {
        // Out of band channels must start with the proxy name, at least for now
        if (strncmp(channelName.c_str(), deviceName.c_str(), deviceName.size()) == 0) {
          channels.push_back(channelName);
        }
      }
      for (auto& channel : channels) {
        LOGP(detail, "Injecting channel '{}' into DPL configuration", channel);
        // Converter should pump messages
        auto& channelPtr = services.get<RawDeviceService>().device()->GetChannel(channel, 0);
        deviceState.inputChannelInfos.push_back(InputChannelInfo{
          .state = InputChannelState::Running,
          .hasPendingEvents = false,
          .readPolled = false,
          .channel = &channelPtr,
          .id = {ChannelIndex::INVALID},
          .channelType = ChannelAccountingType::RAWFMQ,
        });
      }
      numberOfEoS.resize(channels.size(), 0);
      eosPeersCount.resize(channels.size(), 0);
    };

    auto drainMessages = [](ServiceRegistryRef registry, int state) {
      auto* device = registry.get<RawDeviceService>().device();
      auto& deviceState = registry.get<DeviceState>();
      // We drop messages in input only when in ready.
      // FIXME: should we drop messages in input the first time we are in ready?
      if (fair::mq::State{state} != fair::mq::State::Ready) {
        return;
      }
      // We keep track of whether or not all channels have seen a new state.
      std::vector<bool> lastNewStatePending(deviceState.inputChannelInfos.size(), false);
      uv_update_time(deviceState.loop);
      auto start = uv_now(deviceState.loop);

      // Continue iterating until all channels have seen a new state.
      while (std::all_of(lastNewStatePending.begin(), lastNewStatePending.end(), [](bool b) { return b; }) != true) {
        if (uv_now(deviceState.loop) - start > 5000) {
          LOGP(info, "Timeout while draining messages, going to next state anyway.");
          break;
        }
        fair::mq::Parts parts;
        for (size_t ci = 0; ci < deviceState.inputChannelInfos.size(); ++ci) {
          auto& info = deviceState.inputChannelInfos[ci];
          // We only care about rawfmq channels.
          if (info.channelType != ChannelAccountingType::RAWFMQ) {
            lastNewStatePending[ci] = true;
            continue;
          }
          // This means we have not set things up yet. I.e. the first iteration from
          // ready to run has not happened yet.
          if (info.channel == nullptr) {
            lastNewStatePending[ci] = true;
            continue;
          }
          info.channel->Receive(parts, 10);
          // Handle both cases of state changes:
          //
          // - The state has been changed from the outside and FairMQ knows about it.
          // - The state has been changed from the GUI, and deviceState.nextFairMQState knows about it.
          //
          // This latter case is probably better handled from DPL itself, after all it's fair to
          // assume we need to switch state as soon as the GUI notifies us.
          // For now we keep it here to avoid side effects.
          lastNewStatePending[ci] = device->NewStatePending() || (deviceState.nextFairMQState.empty() == false);
          if (parts.Size() == 0) {
            continue;
          }
          if (!lastNewStatePending[ci]) {
            LOGP(warn, "Unexpected {} message on channel {} while in Ready state. Dropping.", parts.Size(), info.channel->GetName());
          } else if (lastNewStatePending[ci]) {
            LOGP(detail, "Some {} parts were received on channel {} while switching away from Ready. Keeping.", parts.Size(), info.channel->GetName());
            for (int pi = 0; pi < parts.Size(); ++pi) {
              info.parts.fParts.emplace_back(std::move(parts.At(pi)));
            }
            info.readPolled = true;
          }
        }
        // Keep state transitions going also when running with the standalone GUI.
        uv_run(deviceState.loop, UV_RUN_NOWAIT);
      }
    };

    ctx.services().get<CallbackService>().set<CallbackService::Id::Start>(channelConfigurationChecker);
    if (ctx.options().get<std::string>("ready-state-policy") == "drain") {
      LOG(info) << "Drain mode requested while in Ready state";
      ctx.services().get<CallbackService>().set<CallbackService::Id::DeviceStateChanged>(drainMessages);
    }

    static auto countEoS = [](fair::mq::Parts& inputs) -> int {
      int count = 0;
      for (int msgidx = 0; msgidx < inputs.Size() / 2; ++msgidx) {
        // Skip when we have nullptr for the header.
        // Not sure it can actually happen, but does not hurt.
        if (inputs.At(msgidx * 2).get() == nullptr) {
          continue;
        }
        auto const sih = o2::header::get<SourceInfoHeader*>(inputs.At(msgidx * 2)->GetData());
        if (sih != nullptr && sih->state == InputChannelState::Completed) {
          count++;
        }
      }
      return count;
    };

    // Data handler for incoming data. Must return true if it sent any data.
    auto dataHandler = [converter, doInjectMissingData, doPrintSizes,
                        outputRoutes = std::move(outputRoutes),
                        control = &ctx.services().get<ControlService>(),
                        deviceState = &ctx.services().get<DeviceState>(),
                        timesliceIndex = &ctx.services().get<TimesliceIndex>(),
                        outputChannels = std::move(outputChannels)](ServiceRegistryRef ref, TimingInfo& timingInfo, fair::mq::Parts& inputs, int, size_t ci, bool newRun) -> bool {
      auto* device = ref.get<RawDeviceService>().device();
      // pass a copy of the outputRoutes
      auto channelRetriever = [&outputRoutes](OutputSpec const& query, DataProcessingHeader::StartTime timeslice) -> std::string {
        for (auto& route : outputRoutes) {
          LOG(debug) << "matching: " << DataSpecUtils::describe(query) << " to route " << DataSpecUtils::describe(route.matcher);
          if (DataSpecUtils::match(route.matcher, query) && ((timeslice % route.maxTimeslices) == route.timeslice)) {
            return route.channel;
          }
        }
        return {""};
      };

      std::string const& channel = channels[ci];
      // we buffer the condition since the converter will forward messages by move
      int nEos = countEoS(inputs);
      if (newRun) {
        std::fill(numberOfEoS.begin(), numberOfEoS.end(), 0);
        std::fill(eosPeersCount.begin(), eosPeersCount.end(), 0);
      }
      numberOfEoS[ci] += nEos;
      if (numberOfEoS[ci]) {
        eosPeersCount[ci] = std::max<int>(eosPeersCount[ci], device->GetNumberOfConnectedPeers(channel));
      }
      // For reference, the oldest possible timeframe passed as newTimesliceId here comes from LifetimeHelpers::enumDrivenCreation()
      bool shouldstop = false;
      if (doInjectMissingData) {
        injectMissingData(*device, inputs, outputRoutes, doInjectMissingData, doPrintSizes);
      }
      bool didSendParts = converter(timingInfo, ref, inputs, channelRetriever, timesliceIndex->getOldestPossibleOutput().timeslice.value, shouldstop);

      // If we have enough EoS messages, we can stop the device
      // Notice that this has a number of failure modes:
      // * If a connection sends the EoS and then closes before the GetNumberOfConnectedPeers command above.
      // * If a connection sends two EoS.
      // * If a connection sends an end of stream closes and another one opens.
      // Finally, if we didn't receive an EoS this time, out counting of the connected peers is off, so the best thing we can do is delay the EoS reporting
      bool everyEoS = shouldstop;
      if (!shouldstop && nEos) {
        everyEoS = true;
        for (unsigned int i = 0; i < numberOfEoS.size(); i++) {
          if (numberOfEoS[i] < eosPeersCount[i]) {
            everyEoS = false;
            break;
          }
        }
      }

      if (everyEoS) {
        LOG(info) << "Received (on channel " << ci << ") " << numberOfEoS[ci] << " end-of-stream from " << eosPeersCount[ci] << " peers, forwarding end-of-stream (shouldstop " << (int)shouldstop << ", nEos " << nEos << ", newRun " << (int)newRun << ")";
        // Mark all input channels as closed
        for (auto& info : deviceState->inputChannelInfos) {
          info.state = InputChannelState::Completed;
        }
        std::fill(numberOfEoS.begin(), numberOfEoS.end(), 0);
        std::fill(eosPeersCount.begin(), eosPeersCount.end(), 0);
        control->endOfStream();
      }
      return didSendParts;
    };

    auto runHandler = [dataHandler, minSHM, sendTFcounter](ProcessingContext& ctx) {
      static RateLimiter limiter;
      static size_t currentRunNumber = -1;
      static bool inStopTransition = false;
      bool newRun = false;
      auto device = ctx.services().get<RawDeviceService>().device();
      if (limiter.check(ctx, std::stoi(device->fConfig->GetValue<std::string>("timeframes-rate-limit")), minSHM)) {
        inStopTransition = true;
      }

      bool didSendParts = false;
      for (size_t ci = 0; ci < channels.size(); ++ci) {
        std::string const& channel = channels[ci];
        int waitTime = channels.size() == 1 ? -1 : 1;
        int maxRead = 1000;
        while (maxRead-- > 0) {
          fair::mq::Parts parts;
          auto res = device->Receive(parts, channel, 0, waitTime);
          if (res == (size_t)fair::mq::TransferCode::error) {
            LOGP(error, "Error while receiving on channel {}", channel);
          }
          // Populate TimingInfo from the first message
          unsigned int nReceived = parts.Size();
          if (nReceived != 0) {
            auto const dh = o2::header::get<DataHeader*>(parts.At(0)->GetData());
            auto& timingInfo = ctx.services().get<TimingInfo>();
            if (dh != nullptr) {
              if (currentRunNumber != -1 && dh->runNumber != 0 && dh->runNumber != currentRunNumber) {
                newRun = true;
                inStopTransition = false;
              }
              if (currentRunNumber == -1 || dh->runNumber != 0) {
                currentRunNumber = dh->runNumber;
              }
              timingInfo.runNumber = dh->runNumber;
              timingInfo.firstTForbit = dh->firstTForbit;
              timingInfo.tfCounter = dh->tfCounter;
            }
            auto const dph = o2::header::get<DataProcessingHeader*>(parts.At(0)->GetData());
            if (dph != nullptr) {
              timingInfo.timeslice = dph->startTime;
              timingInfo.creation = dph->creation;
            }
            if (!inStopTransition) {
              didSendParts |= dataHandler(ctx.services(), timingInfo, parts, 0, ci, newRun);
            }
            if (sendTFcounter) {
              ctx.services().get<o2::monitoring::Monitoring>().send(o2::monitoring::Metric{(uint64_t)timingInfo.tfCounter, "df-sent"}.addTag(o2::monitoring::tags::Key::Subsystem, o2::monitoring::tags::Value::DPL));
            }
          }
          if (nReceived == 0 || channels.size() == 1) {
            break;
          }
          waitTime = 0;
        }
      }
      // In case we did not send any part at all, we need to rewind by one
      // to avoid creating extra timeslices.
      auto& decongestion = ctx.services().get<DecongestionService>();
      decongestion.nextEnumerationTimesliceRewinded = !didSendParts;
      if (didSendParts) {
        ctx.services().get<MessageContext>().fakeDispatch();
      } else {
        decongestion.nextEnumerationTimeslice -= 1;
      }
    };

    return runHandler;
  }};
  const char* d = strdup(((std::string(defaultChannelConfig).find("name=") == std::string::npos ? (std::string("name=") + name + ",") : "") + std::string(defaultChannelConfig)).c_str());
  spec.options = {
    ConfigParamSpec{"ready-state-policy", VariantType::String, "keep", {"What to do when the device is in ready state: *keep*, drain"}},
    ConfigParamSpec{"channel-config", VariantType::String, d, {"Out-of-band channel config"}}};
  return spec;
}

// Decide where to sent the output. Everything to "downstream" if there is such a channel.
std::string defaultOutputProxyChannelSelector(InputSpec const& input, const std::unordered_map<std::string, std::vector<fair::mq::Channel>>& channels)
{
  return channels.count("downstream") ? "downstream" : input.binding;
}

DataProcessorSpec specifyFairMQDeviceOutputProxy(char const* name,
                                                 Inputs const& inputSpecs,
                                                 const char* defaultChannelConfig)
{
  DataProcessorSpec spec;
  spec.name = name;
  spec.inputs = inputSpecs;
  spec.outputs = {};
  spec.algorithm = adaptStateful([inputSpecs](FairMQDeviceProxy& proxy, CallbackService& callbacks, RawDeviceService& rds, DeviceSpec const& deviceSpec, ConfigParamRegistry const& options) {
    // we can retrieve the channel name from the channel configuration string
    // FIXME: even if a --channel-config option is specified on the command line, always the default string
    // is retrieved from the config registry. The channel name thus needs to be configured in the default
    // string AND must match the name in an optional channel config.
    auto channelConfig = options.get<std::string>("channel-config");
    std::regex r{R"(name=([^,]*))"};
    std::vector<std::string> values{std::sregex_token_iterator{std::begin(channelConfig), std::end(channelConfig), r, 1},
                                    std::sregex_token_iterator{}};
    if (values.size() != 1 || values[0].empty()) {
      throw std::runtime_error("failed to extract channel name from channel configuration parameter '" + channelConfig + "'");
    }
    std::string outputChannelName = values[0];

    auto* device = rds.device();
    // check that the input spec bindings have corresponding output channels
    // fair::mq::Device calls the custom init before the channels have been configured
    // so we do the check before starting in a dedicated callback
    auto channelConfigurationChecker = [inputSpecs = std::move(inputSpecs), device, outputChannelName]() {
      LOG(info) << "checking channel configuration";
      if (device->GetChannels().count(outputChannelName) == 0) {
        throw std::runtime_error("no corresponding output channel found for input '" + outputChannelName + "'");
      }
    };
    callbacks.set<CallbackService::Id::Start>(channelConfigurationChecker);
    auto lastDataProcessingHeader = std::make_shared<DataProcessingHeader>(0, 0);

    auto& spec = const_cast<DeviceSpec&>(deviceSpec);
    for (auto const& inputSpec : inputSpecs) {
      // this is a prototype, in principle we want to have all spec objects const
      // and so only the const object can be retrieved from service registry
      ForwardRoute route{0, 1, inputSpec, outputChannelName};
      spec.forwards.emplace_back(route);
    }

    auto forwardEos = [device, lastDataProcessingHeader, outputChannelName](EndOfStreamContext&) {
      // DPL implements an internal end of stream signal, which is propagated through
      // all downstream channels if a source is dry, make it available to other external
      // devices via a message of type {DPL/EOS/0}
      for (auto& channelInfo : device->GetChannels()) {
        auto& channelName = channelInfo.first;
        if (channelName != outputChannelName) {
          continue;
        }
        DataHeader dh;
        dh.dataOrigin = "DPL";
        dh.dataDescription = "EOS";
        dh.subSpecification = 0;
        dh.payloadSize = 0;
        dh.payloadSerializationMethod = o2::header::gSerializationMethodNone;
        dh.tfCounter = 0;
        dh.firstTForbit = 0;
        SourceInfoHeader sih;
        sih.state = InputChannelState::Completed;
        // allocate the header message using the underlying transport of the channel
        auto channelAlloc = o2::pmr::getTransportAllocator(channelInfo.second[0].Transport());
        auto headerMessage = o2::pmr::getMessage(o2::header::Stack{channelAlloc, dh, *lastDataProcessingHeader, sih});
        fair::mq::Parts out;
        out.AddPart(std::move(headerMessage));
        // add empty payload message
        out.AddPart(device->NewMessageFor(channelName, 0, 0));
        sendOnChannel(*device, out, channelName, (size_t)-1);
      }
    };
    callbacks.set<CallbackService::Id::EndOfStream>(forwardEos);

    return adaptStateless([lastDataProcessingHeader](InputRecord& inputs) {
      for (size_t ii = 0; ii != inputs.size(); ++ii) {
        for (size_t pi = 0; pi < inputs.getNofParts(ii); ++pi) {
          auto part = inputs.getByPos(ii, pi);
          const auto* dph = o2::header::get<DataProcessingHeader*>(part.header);
          if (dph) {
            // FIXME: should we implement an assignment operator for DataProcessingHeader?
            lastDataProcessingHeader->startTime = dph->startTime;
            lastDataProcessingHeader->duration = dph->duration;
            lastDataProcessingHeader->creation = dph->creation;
          }
        }
      }
    });
  });
  const char* d = strdup(((std::string(defaultChannelConfig).find("name=") == std::string::npos ? (std::string("name=") + name + ",") : "") + std::string(defaultChannelConfig)).c_str());
  spec.options = {
    ConfigParamSpec{"channel-config", VariantType::String, d, {"Out-of-band channel config"}},
  };

  return spec;
}

DataProcessorSpec specifyFairMQDeviceMultiOutputProxy(char const* name,
                                                      Inputs const& inputSpecs,
                                                      const char* defaultChannelConfig,
                                                      ChannelSelector channelSelector)
{
  // FIXME: this looks like a code duplication with the function above, check if the
  // two can be combined
  DataProcessorSpec spec;
  spec.name = name;
  spec.inputs = inputSpecs;
  spec.outputs = {};
  spec.algorithm = adaptStateful([inputSpecs, channelSelector](FairMQDeviceProxy& proxy, CallbackService& callbacks, RawDeviceService& rds, const DeviceSpec& deviceSpec) {
    auto device = rds.device();
    // check that the input spec bindings have corresponding output channels
    // fair::mq::Device calls the custom init before the channels have been configured
    // so we do the check before starting in a dedicated callback
    // also we set forwards for all input specs and keep a list of all channels so we can send EOS on them
    auto channelNames = std::make_shared<std::vector<std::string>>();
    auto channelConfigurationInitializer = [&proxy, inputSpecs = std::move(inputSpecs), device, channelSelector, &deviceSpec, channelNames]() {
      channelNames->clear();
      auto& mutableDeviceSpec = const_cast<DeviceSpec&>(deviceSpec);
      for (auto const& spec : inputSpecs) {
        auto channel = channelSelector(spec, device->GetChannels());
        if (device->GetChannels().count(channel) == 0) {
          throw std::runtime_error("no corresponding output channel found for input '" + channel + "'");
        }
        ForwardRoute route{0, 1, spec, channel};
        // this we will try to fix on the framework level, there will be an API to
        // set external routes. Basically, this has to be added while setting up the
        // workflow. After that, the actual spec provided by the service is supposed
        // to be const by design
        mutableDeviceSpec.forwards.emplace_back(route);

        channelNames->emplace_back(std::move(channel));
      }
      proxy.bind(mutableDeviceSpec.outputs, mutableDeviceSpec.inputs, mutableDeviceSpec.forwards, *device);
    };
    // We need to clear the channels on stop, because we will check and add them
    auto channelConfigurationDisposer = [&deviceSpec]() {
      auto& mutableDeviceSpec = const_cast<DeviceSpec&>(deviceSpec);
      mutableDeviceSpec.forwards.clear();
    };
    callbacks.set<CallbackService::Id::Start>(channelConfigurationInitializer);
    callbacks.set<CallbackService::Id::Stop>(channelConfigurationDisposer);

    auto lastDataProcessingHeader = std::make_shared<DataProcessingHeader>(0, 0);
    auto forwardEos = [device, lastDataProcessingHeader, channelNames](EndOfStreamContext&) {
      // DPL implements an internal end of stream signal, which is propagated through
      // all downstream channels if a source is dry, make it available to other external
      // devices via a message of type {DPL/EOS/0}
      for (auto& channelInfo : device->GetChannels()) {
        auto& channelName = channelInfo.first;
        auto checkChannel = [channelNames = std::move(*channelNames)](std::string const& name) -> bool {
          for (auto const& n : channelNames) {
            if (n == name) {
              return true;
            }
          }
          return false;
        };
        if (!checkChannel(channelName)) {
          continue;
        }
        DataHeader dh;
        dh.dataOrigin = "DPL";
        dh.dataDescription = "EOS";
        dh.subSpecification = 0;
        dh.payloadSize = 0;
        dh.payloadSerializationMethod = o2::header::gSerializationMethodNone;
        dh.tfCounter = 0;
        dh.firstTForbit = 0;
        SourceInfoHeader sih;
        sih.state = InputChannelState::Completed;
        // allocate the header message using the underlying transport of the channel
        auto channelAlloc = o2::pmr::getTransportAllocator(channelInfo.second[0].Transport());
        auto headerMessage = o2::pmr::getMessage(o2::header::Stack{channelAlloc, dh, *lastDataProcessingHeader, sih});
        fair::mq::Parts out;
        out.AddPart(std::move(headerMessage));
        // add empty payload message
        out.AddPart(device->NewMessageFor(channelName, 0, 0));
        LOGP(detail, "Forwarding EoS to {}", channelName);
        sendOnChannel(*device, out, channelName, (size_t)-1);
      }
    };
    callbacks.set<CallbackService::Id::EndOfStream>(forwardEos);

    return adaptStateless([channelSelector, lastDataProcessingHeader](InputRecord& inputs) {
      // there is nothing to do if the forwarding is handled on the framework level
      // as forward routes but we need to keep a copy of the last DataProcessingHeader
      // for sending the EOS
      for (size_t ii = 0; ii != inputs.size(); ++ii) {
        for (size_t pi = 0; pi < inputs.getNofParts(ii); ++pi) {
          auto part = inputs.getByPos(ii, pi);
          const auto* dph = o2::header::get<DataProcessingHeader*>(part.header);
          if (dph) {
            // FIXME: should we implement an assignment operator for DataProcessingHeader?
            lastDataProcessingHeader->startTime = dph->startTime;
            lastDataProcessingHeader->duration = dph->duration;
            lastDataProcessingHeader->creation = dph->creation;
          }
        }
      }
    });
  });
  const char* d = strdup(((std::string(defaultChannelConfig).find("name=") == std::string::npos ? (std::string("name=") + name + ",") : "") + std::string(defaultChannelConfig)).c_str());
  spec.options = {
    ConfigParamSpec{"channel-config", VariantType::String, d, {"Out-of-band channel config"}},
  };

  return spec;
}

} // namespace o2::framework
