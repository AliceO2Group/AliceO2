// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "Framework/DataProcessingHeader.h"
#include "Framework/InputSpec.h"
#include "Framework/LifetimeHelpers.h"
#include "Framework/Logger.h"
#include "Framework/RawDeviceService.h"
#include "Framework/ServiceRegistry.h"
#include "Framework/TimesliceIndex.h"

#include "Headers/DataHeader.h"
#include "Headers/Stack.h"
#include "MemoryResources/MemoryResources.h"
#include <curl/curl.h>

#include <fairmq/FairMQDevice.h>

#include <cstdlib>

using namespace o2::header;
using namespace fair;

namespace o2
{
namespace framework
{

namespace
{
size_t getCurrentTime()
{
  auto now = std::chrono::system_clock::now();
  auto duration = now.time_since_epoch();
  return std::chrono::duration_cast<std::chrono::microseconds>(duration).count();
}
} // namespace

ExpirationHandler::Creator LifetimeHelpers::dataDrivenCreation()
{
  return [](TimesliceIndex&) -> TimesliceSlot {
    return {TimesliceSlot::INVALID};
  };
}

ExpirationHandler::Creator LifetimeHelpers::enumDrivenCreation(size_t start, size_t end, size_t step)
{
  auto last = std::make_shared<size_t>(start);
  return [start, end, step, last](TimesliceIndex& index) -> TimesliceSlot {
    for (size_t si = 0; si < index.size(); si++) {
      if (*last > end) {
        return TimesliceSlot{TimesliceSlot::INVALID};
      }
      auto slot = TimesliceSlot{si};
      if (index.isValid(slot) == false) {
        TimesliceId timestamp{*last};
        *last += step;
        index.associate(timestamp, slot);
        return slot;
      }
    }
    return TimesliceSlot{TimesliceSlot::INVALID};
  };
}

ExpirationHandler::Creator LifetimeHelpers::timeDrivenCreation(std::chrono::microseconds period)
{
  auto start = getCurrentTime();
  auto last = std::make_shared<decltype(start)>(start);
  // FIXME: should create timeslices when period expires....
  return [last, period](TimesliceIndex& index) -> TimesliceSlot {
    // Nothing to do if the time has not expired yet.
    auto current = getCurrentTime();
    auto delta = current - *last;
    if (delta < period.count()) {
      return TimesliceSlot{TimesliceSlot::INVALID};
    }
    // We first check if the current time is not already present
    // FIXME: this should really be done by query matching? Ok
    //        for now to avoid duplicate entries.
    for (size_t i = 0; i < index.size(); ++i) {
      TimesliceSlot slot{i};
      if (index.isValid(slot) == false) {
        continue;
      }
      if (index.getTimesliceForSlot(slot).value == current) {
        return TimesliceSlot{TimesliceSlot::INVALID};
      }
    }
    // If we are here the timer has expired and a new slice needs
    // to be created.
    *last = current;
    data_matcher::VariableContext newContext;
    newContext.put({0, static_cast<uint64_t>(current)});
    newContext.commit();
    auto [action, slot] = index.replaceLRUWith(newContext);
    switch (action) {
      case TimesliceIndex::ActionTaken::ReplaceObsolete:
      case TimesliceIndex::ActionTaken::ReplaceUnused:
        index.associate(TimesliceId{current}, slot);
        break;
      case TimesliceIndex::ActionTaken::DropInvalid:
      case TimesliceIndex::ActionTaken::DropObsolete:
        break;
    }
    return slot;
  };
}

ExpirationHandler::Checker LifetimeHelpers::expireNever()
{
  return [](int64_t) -> bool { return false; };
}

ExpirationHandler::Checker LifetimeHelpers::expireAlways()
{
  return [](int64_t) -> bool { return true; };
}

ExpirationHandler::Checker LifetimeHelpers::expireTimed(std::chrono::microseconds period)
{
  auto start = getCurrentTime();
  auto last = std::make_shared<decltype(start)>(start);
  return [last, period](int64_t) -> bool {
    auto current = getCurrentTime();
    auto delta = current - *last;
    if (delta > period.count()) {
      *last = current;
      return true;
    }
    return false;
  };
}

/// Does nothing. Use this for cases where you do not want to do anything
/// when records expire. This is the default behavior for data (which never
/// expires via this mechanism).
ExpirationHandler::Handler LifetimeHelpers::doNothing()
{
  return [](ServiceRegistry&, PartRef& ref, uint64_t) -> void { return; };
}

// We simply put everything in a stringstream and read it afterwards.
size_t readToMessage(void* p, size_t size, size_t nmemb, void* userdata)
{
  if (nmemb == 0) {
    return 0;
  }
  if (size == 0) {
    return 0;
  }
  o2::vector<char>* buffer = (o2::vector<char>*)userdata;
  size_t oldSize = buffer->size();
  buffer->resize(oldSize + nmemb * size);
  memcpy(buffer->data() + oldSize, userdata, nmemb * size);
  return size * nmemb;
}

/// Fetch an object from CCDB if the record is expired. The actual
/// name of the object is given by:
///
/// "<namespace>/<InputRoute.origin>/<InputRoute.description>"
///
/// \todo for the moment we always go to CCDB every time we are expired.
/// \todo this should really be done in the common fetcher.
/// \todo provide a way to customize the namespace from the ProcessingContext
ExpirationHandler::Handler
  LifetimeHelpers::fetchFromCCDBCache(ConcreteDataMatcher const& matcher,
                                      std::string const& prefix,
                                      std::string const& overrideTimestamp,
                                      std::string const& sourceChannel)
{
  char* err;
  uint64_t overrideTimestampMilliseconds = strtoll(overrideTimestamp.c_str(), &err, 10);
  if (*err != 0) {
    throw std::runtime_error("fetchFromCCDBCache: Unable to parse forced timestamp for conditions");
  }
  if (overrideTimestampMilliseconds) {
    LOGP(info, "fetchFromCCDBCache: forcing timestamp for conditions to {} milliseconds from epoch UTC", overrideTimestampMilliseconds);
  }
  return [matcher, sourceChannel, serverUrl = prefix, overrideTimestampMilliseconds](ServiceRegistry& services, PartRef& ref, uint64_t timestamp) -> void {
    // We should invoke the handler only once.
    assert(!ref.header);
    assert(!ref.payload);

    auto& rawDeviceService = services.get<RawDeviceService>();
    auto&& transport = rawDeviceService.device()->GetChannel(sourceChannel, 0).Transport();
    auto channelAlloc = o2::pmr::getTransportAllocator(transport);
    o2::vector<char> payloadBuffer;
    payloadBuffer.reserve(10000); // we begin with messages of 10KB
    auto payload = o2::pmr::getMessage(std::forward<o2::vector<char>>(payloadBuffer), transport->GetMemoryResource());

    CURL* curl = curl_easy_init();
    if (curl == nullptr) {
      throw std::runtime_error("fetchFromCCDBCache: Unable to initialise CURL");
    }
    CURLcode res;
    if (overrideTimestampMilliseconds) {
      timestamp = overrideTimestampMilliseconds;
    }
    auto path = std::string("/") + matcher.origin.as<std::string>() + "/" + matcher.description.as<std::string>() + "/" + std::to_string(timestamp / 1000);
    auto url = serverUrl + path;
    LOG(INFO) << "fetchFromCCDBCache: Fetching " << url;

    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &payloadBuffer);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, readToMessage);

    res = curl_easy_perform(curl);
    if (res != CURLE_OK) {
      throw std::runtime_error(std::string("fetchFromCCDBCache: Unable to fetch ") + url + " from CCDB");
    }
    long responseCode;
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &responseCode);

    if (responseCode != 200) {
      throw std::runtime_error(std::string("fetchFromCCDBCache: HTTP error ") + std::to_string(responseCode) + " while fetching " + url + " from CCDB");
    }

    curl_easy_cleanup(curl);

    DataHeader dh;
    dh.dataOrigin = matcher.origin;
    dh.dataDescription = matcher.description;
    dh.subSpecification = matcher.subSpec;
    // FIXME: should use curl_off_t and CURLINFO_SIZE_DOWNLOAD_T, but
    //        apparently not there on some platforms.
    double dl;
    res = curl_easy_getinfo(curl, CURLINFO_SIZE_DOWNLOAD, &dl);
    dh.payloadSize = payloadBuffer.size();
    dh.payloadSerializationMethod = gSerializationMethodNone;

    DataProcessingHeader dph{timestamp, 1};
    auto header = o2::pmr::getMessage(o2::header::Stack{channelAlloc, dh, dph});

    ref.header = std::move(header);
    ref.payload = std::move(payload);

    return;
  };
}

/// Create an entry in the registry for histograms on the first
/// FIXME: actually implement this
/// FIXME: provide a way to customise the histogram from the configuration.
ExpirationHandler::Handler LifetimeHelpers::fetchFromQARegistry()
{
  return [](ServiceRegistry&, PartRef& ref, uint64_t) -> void {
    throw std::runtime_error("fetchFromQARegistry: Not yet implemented");
    return;
  };
}

/// Create an entry in the registry for histograms on the first
/// FIXME: actually implement this
/// FIXME: provide a way to customise the histogram from the configuration.
ExpirationHandler::Handler LifetimeHelpers::fetchFromObjectRegistry()
{
  return [](ServiceRegistry&, PartRef& ref, uint64_t) -> void {
    throw std::runtime_error("fetchFromObjectRegistry: Not yet implemented");
    return;
  };
}

/// Enumerate entries on every invokation.
ExpirationHandler::Handler LifetimeHelpers::enumerate(ConcreteDataMatcher const& matcher, std::string const& sourceChannel)
{
  auto counter = std::make_shared<int64_t>(0);
  auto f = [matcher, counter, sourceChannel](ServiceRegistry& services, PartRef& ref, uint64_t timestamp) -> void {
    // We should invoke the handler only once.
    assert(!ref.header);
    assert(!ref.payload);
    auto& rawDeviceService = services.get<RawDeviceService>();

    DataHeader dh;
    dh.dataOrigin = matcher.origin;
    dh.dataDescription = matcher.description;
    dh.subSpecification = matcher.subSpec;
    dh.payloadSize = 8;
    dh.payloadSerializationMethod = gSerializationMethodNone;

    DataProcessingHeader dph{timestamp, 1};

    auto&& transport = rawDeviceService.device()->GetChannel(sourceChannel, 0).Transport();
    auto channelAlloc = o2::pmr::getTransportAllocator(transport);
    auto header = o2::pmr::getMessage(o2::header::Stack{channelAlloc, dh, dph});
    ref.header = std::move(header);

    auto payload = rawDeviceService.device()->NewMessage(*counter);
    ref.payload = std::move(payload);
    (*counter)++;
  };
  return f;
}

} // namespace framework
} // namespace o2
