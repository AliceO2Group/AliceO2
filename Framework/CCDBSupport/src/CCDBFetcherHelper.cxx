// Copyright 2019-2025 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#include "CCDBFetcherHelper.h"
#include "Framework/DataTakingContext.h"
#include "Framework/Signpost.h"
#include "Framework/DataSpecUtils.h"
#include "Framework/ConfigParamRegistry.h"
#include <TError.h>
#include <TMemFile.h>

O2_DECLARE_DYNAMIC_LOG(ccdb);

namespace o2::framework
{

o2::ccdb::CcdbApi& CCDBFetcherHelper::getAPI(const std::string& path)
{
  // find the first = sign in the string. If present drop everything after it
  // and between it and the previous /.
  auto pos = path.find('=');
  if (pos == std::string::npos) {
    auto entry = remappings.find(path);
    return apis[entry == remappings.end() ? "" : entry->second];
  }
  auto pos2 = path.rfind('/', pos);
  if (pos2 == std::string::npos || pos2 == pos - 1 || pos2 == 0) {
    throw runtime_error_f("Malformed path %s", path.c_str());
  }
  auto entry = remappings.find(path.substr(0, pos2));
  return apis[entry == remappings.end() ? "" : entry->second];
}

namespace
{
bool isOnlineRun(DataTakingContext const& dtc)
{
  return dtc.deploymentMode == DeploymentMode::OnlineAUX || dtc.deploymentMode == DeploymentMode::OnlineDDS || dtc.deploymentMode == DeploymentMode::OnlineECS;
}
} // namespace

void CCDBFetcherHelper::initialiseHelper(CCDBFetcherHelper& helper, ConfigParamRegistry const& options)
{
  auto defHost = options.get<std::string>("condition-backend");
  auto checkRate = options.get<int>("condition-tf-per-query");
  auto checkMult = options.get<int>("condition-tf-per-query-multiplier");
  helper.useTFSlice = options.get<int>("condition-use-slice-for-prescaling");
  helper.timeToleranceMS = options.get<int64_t>("condition-time-tolerance");
  helper.queryPeriodGlo = checkRate > 0 ? checkRate : std::numeric_limits<int>::max();
  helper.queryPeriodFactor = checkMult == 0 ? 1 : checkMult;
  std::string extraCond{};
  if (helper.useTFSlice) {
    extraCond = ". Use TFSlice";
    if (helper.useTFSlice > 0) {
      extraCond += fmt::format(" + max TFcounter jump <= {}", helper.useTFSlice);
    }
  }
  LOGP(info, "CCDB Backend at: {}, validity check for every {} TF{}{}", defHost, helper.queryPeriodGlo,
       helper.queryPeriodFactor == 1 ? std::string{} : (helper.queryPeriodFactor > 0 ? fmt::format(", (query for high-rate objects downscaled by {})", helper.queryPeriodFactor) : fmt::format(", (query downscaled as TFcounter%{})", -helper.queryPeriodFactor)),
       extraCond);
  LOGP(info, "Hook to enable signposts for CCDB messages at {}", (void*)&private_o2_log_ccdb->stacktrace);
  auto remapString = options.get<std::string>("condition-remap");
  ParserResult result = parseRemappings(remapString.c_str());
  if (!result.error.empty()) {
    throw runtime_error_f("Error while parsing remapping string %s", result.error.c_str());
  }
  helper.remappings = result.remappings;
  helper.apis[""].init(defHost); // default backend
  LOGP(info, "Initialised default CCDB host {}", defHost);
  //
  for (auto& entry : helper.remappings) { // init api instances for every host seen in the remapping
    if (helper.apis.find(entry.second) == helper.apis.end()) {
      helper.apis[entry.second].init(entry.second);
      LOGP(info, "Initialised custom CCDB host {}", entry.second);
    }
    LOGP(info, "{} is remapped to {}", entry.first, entry.second);
  }
  helper.createdNotBefore = std::to_string(options.get<int64_t>("condition-not-before"));
  helper.createdNotAfter = std::to_string(options.get<int64_t>("condition-not-after"));
}

CCDBFetcherHelper::ParserResult CCDBFetcherHelper::parseRemappings(char const* str)
{
  std::unordered_map<std::string, std::string> remappings;
  std::string currentUrl = "";

  enum ParsingStates {
    IN_BEGIN,
    IN_BEGIN_URL,
    IN_BEGIN_TARGET,
    IN_END_TARGET,
    IN_END_URL
  };
  ParsingStates state = IN_BEGIN;

  while (true) {
    switch (state) {
      case IN_BEGIN: {
        if (*str == 0) {
          return {remappings, ""};
        }
        state = IN_BEGIN_URL;
      }
      case IN_BEGIN_URL: {
        if ((strncmp("http://", str, 7) != 0) && (strncmp("https://", str, 8) != 0 && (strncmp("file://", str, 7) != 0))) {
          return {remappings, "URL should start with either http:// or https:// or file://"};
        }
        state = IN_END_URL;
      } break;
      case IN_END_URL: {
        char const* c = strchr(str, '=');
        if (c == nullptr) {
          return {remappings, "Expecting at least one target path, missing `='?"};
        }
        if ((c - str) == 0) {
          return {remappings, "Empty url"};
        }
        currentUrl = std::string_view(str, c - str);
        state = IN_BEGIN_TARGET;
        str = c + 1;
      } break;
      case IN_BEGIN_TARGET: {
        if (*str == 0) {
          return {remappings, "Empty target"};
        }
        state = IN_END_TARGET;
      } break;
      case IN_END_TARGET: {
        char const* c = strpbrk(str, ",;");
        if (c == nullptr) {
          if (remappings.count(str)) {
            return {remappings, fmt::format("Path {} requested more than once.", str)};
          }
          remappings[std::string(str)] = currentUrl;
          return {remappings, ""};
        }
        if ((c - str) == 0) {
          return {remappings, "Empty target"};
        }
        auto key = std::string(str, c - str);
        if (remappings.count(str)) {
          return {remappings, fmt::format("Path {} requested more than once.", key)};
        }
        remappings[key] = currentUrl;
        if (*c == ';') {
          state = IN_BEGIN_URL;
        } else {
          state = IN_BEGIN_TARGET;
        }
        str = c + 1;
      } break;
    }
  }
}

auto CCDBFetcherHelper::populateCacheWith(std::shared_ptr<CCDBFetcherHelper> const& helper,
                                          std::vector<CCDBFetcherHelper::FetchOp> const& ops,
                                          TimingInfo& timingInfo,
                                          DataTakingContext& dtc,
                                          DataAllocator& allocator) -> std::vector<CCDBFetcherHelper::Response>
{
  int objCnt = -1;
  // We use the timeslice, so that we hook into the same interval as the rest of the
  // callback.
  static bool isOnline = isOnlineRun(dtc);

  auto sid = _o2_signpost_id_t{(int64_t)timingInfo.timeslice};
  O2_SIGNPOST_START(ccdb, sid, "populateCacheWith", "Starting to populate cache with CCDB objects");
  std::vector<Response> responses;
  for (auto& op : ops) {
    int64_t timestampToUse = op.timestamp;
    O2_SIGNPOST_EVENT_EMIT(ccdb, sid, "populateCacheWith", "Fetching object for route %{public}s", DataSpecUtils::describe(op.spec).data());
    objCnt++;
    auto concrete = DataSpecUtils::asConcreteDataMatcher(op.spec);
    Output output{concrete.origin, concrete.description, concrete.subSpec};
    auto&& v = allocator.makeVector<char>(output);
    std::map<std::string, std::string> metadata;
    std::map<std::string, std::string> headers;
    std::string path = op.url;
    std::string etag = "";
    int chRate = helper->queryPeriodGlo;
    bool checkValidity = false;
    if (op.runDependent > 0) {
      if (op.runDependent == 1) {
        metadata["runNumber"] = std::format("{}", op.runNumber);
      } else if (op.runDependent == 2) {
        timestampToUse = op.runNumber;
      } else {
        LOGP(fatal, "Undefined ccdb-run-dependent option {} for spec {}/{}/{}", op.runDependent,
             concrete.origin.as<std::string>(), concrete.description.as<std::string>(), int(concrete.subSpec));
      }
    }
    for (auto m : op.metadata) {
      O2_SIGNPOST_EVENT_EMIT(ccdb, sid, "populateCacheWith", "Adding metadata %{public}s: %{public}s to the request", m.key.data(), m.value.data());
      metadata[m.key] = m.value;
    }
    if (op.queryRate != 0) {
      chRate = op.queryRate * helper->queryPeriodFactor;
    }

    const auto url2uuid = helper->mapURL2UUID.find(path);
    if (url2uuid != helper->mapURL2UUID.end()) {
      etag = url2uuid->second.etag;
      // We check validity every chRate timeslices or if the cache is expired
      uint64_t validUntil = url2uuid->second.cacheValidUntil;
      // When the cache was populated. If the cache was populated after the timestamp, we need to check validity.
      uint64_t cachePopulatedAt = url2uuid->second.cachePopulatedAt;
      // If timestamp is before the time the element was cached or after the claimed validity, we need to check validity, again
      // when online.
      bool cacheExpired = (validUntil <= timestampToUse) || (op.timestamp < cachePopulatedAt);
      if (isOnline || cacheExpired) {
        if (!helper->useTFSlice) {
          checkValidity = chRate > 0 ? (std::abs(int(timingInfo.tfCounter - url2uuid->second.lastCheckedTF)) >= chRate) : (timingInfo.tfCounter % -chRate) == 0;
        } else {
          checkValidity = chRate > 0 ? (std::abs(int(timingInfo.timeslice - url2uuid->second.lastCheckedSlice)) >= chRate) : (timingInfo.timeslice % -chRate) == 0;
          if (!checkValidity && helper->useTFSlice > std::abs(chRate)) { // make sure the interval is tolerated unless the check rate itself is too large
            checkValidity = std::abs(int(timingInfo.tfCounter) - url2uuid->second.lastCheckedTF) > helper->useTFSlice;
          }
        }
      }
    } else {
      checkValidity = true; // never skip check if the cache is empty
    }

    O2_SIGNPOST_EVENT_EMIT(ccdb, sid, "populateCacheWith", "checkValidity is %{public}s for tf%{public}s %zu of %{public}s", checkValidity ? "true" : "false", helper->useTFSlice ? "ID" : "Slice", helper->useTFSlice ? timingInfo.timeslice : timingInfo.tfCounter, path.data());

    const auto& api = helper->getAPI(path);
    if (checkValidity && (!api.isSnapshotMode() || etag.empty())) { // in the snapshot mode the object needs to be fetched only once
      LOGP(detail, "Loading {} for timestamp {}", path, timestampToUse);
      api.loadFileToMemory(v, path, metadata, timestampToUse, &headers, etag, helper->createdNotAfter, helper->createdNotBefore);
      if ((headers.count("Error") != 0) || (etag.empty() && v.empty())) {
        LOGP(fatal, "Unable to find CCDB object {}/{}", path, timestampToUse);
        // FIXME: I should send a dummy message.
        continue;
      }
      // printing in case we find a default entry
      if (headers.find("default") != headers.end()) {
        LOGP(detail, "******** Default entry used for {} ********", path);
      }
      helper->mapURL2UUID[path].lastCheckedTF = timingInfo.tfCounter;
      helper->mapURL2UUID[path].lastCheckedSlice = timingInfo.timeslice;
      if (etag.empty()) {
        helper->mapURL2UUID[path].etag = headers["ETag"]; // update uuid
        helper->mapURL2UUID[path].cachePopulatedAt = timestampToUse;
        helper->mapURL2UUID[path].cacheMiss++;
        helper->mapURL2UUID[path].size = v.size();
        helper->mapURL2UUID[path].minSize = std::min(v.size(), helper->mapURL2UUID[path].minSize);
        helper->mapURL2UUID[path].maxSize = std::max(v.size(), helper->mapURL2UUID[path].maxSize);
        auto size = v.size();
        helper->totalFetchedBytes += size;
        helper->totalRequestedBytes += size;
        api.appendFlatHeader(v, headers);
        auto cacheId = allocator.adoptContainer(output, std::move(v), DataAllocator::CacheStrategy::Always, header::gSerializationMethodCCDB);
        helper->mapURL2DPLCache[path] = cacheId;
        responses.emplace_back(Response{.id = cacheId, .size = size, .request = nullptr});
        O2_SIGNPOST_EVENT_EMIT(ccdb, sid, "populateCacheWith", "Caching %{public}s for %{public}s (DPL id %" PRIu64 ", size %zu)", path.data(), headers["ETag"].data(), cacheId.value, size);
        continue;
      }
      if (v.size()) { // but should be overridden by fresh object
        // somewhere here pruneFromCache should be called
        helper->mapURL2UUID[path].etag = headers["ETag"]; // update uuid
        helper->mapURL2UUID[path].cachePopulatedAt = timestampToUse;
        helper->mapURL2UUID[path].cacheValidUntil = headers["Cache-Valid-Until"].empty() ? 0 : std::stoul(headers["Cache-Valid-Until"]);
        helper->mapURL2UUID[path].cacheMiss++;
        helper->mapURL2UUID[path].size = v.size();
        helper->mapURL2UUID[path].minSize = std::min(v.size(), helper->mapURL2UUID[path].minSize);
        helper->mapURL2UUID[path].maxSize = std::max(v.size(), helper->mapURL2UUID[path].maxSize);
        auto size = v.size();
        helper->totalFetchedBytes += size;
        helper->totalRequestedBytes += size;
        api.appendFlatHeader(v, headers);
        auto cacheId = allocator.adoptContainer(output, std::move(v), DataAllocator::CacheStrategy::Always, header::gSerializationMethodCCDB);
        helper->mapURL2DPLCache[path] = cacheId;
        responses.emplace_back(Response{.id = cacheId, .size = size, .request = nullptr});
        O2_SIGNPOST_EVENT_EMIT(ccdb, sid, "populateCacheWith", "Caching %{public}s for %{public}s (DPL id %" PRIu64 ")", path.data(), headers["ETag"].data(), cacheId.value);
        // one could modify the    adoptContainer to take optional old cacheID to clean:
        // mapURL2DPLCache[URL] = ctx.outputs().adoptContainer(output, std::move(outputBuffer), DataAllocator::CacheStrategy::Always, mapURL2DPLCache[URL]);
        continue;
      } else {
        // Only once the etag is actually used, we get the information on how long the object is valid
        helper->mapURL2UUID[path].cacheValidUntil = headers["Cache-Valid-Until"].empty() ? 0 : std::stoul(headers["Cache-Valid-Until"]);
      }
    }
    // cached object is fine
    auto cacheId = helper->mapURL2DPLCache[path];
    O2_SIGNPOST_EVENT_EMIT(ccdb, sid, "populateCacheWith", "Reusing %{public}s for %{public}s (DPL id %" PRIu64 ")", path.data(), headers["ETag"].data(), cacheId.value);
    helper->mapURL2UUID[path].cacheHit++;
    responses.emplace_back(Response{.id = cacheId, .size = helper->mapURL2UUID[path].size, .request = nullptr});
    allocator.adoptFromCache(output, cacheId, header::gSerializationMethodCCDB);
    // the outputBuffer was not used, can we destroy it?
  }
  O2_SIGNPOST_END(ccdb, sid, "populateCacheWith", "Finished populating cache with CCDB objects");
  return responses;
};

} // namespace o2::framework
