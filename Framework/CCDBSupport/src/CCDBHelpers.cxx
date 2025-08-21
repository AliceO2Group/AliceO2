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

#include "CCDBHelpers.h"
#include "CCDBFetcherHelper.h"
#include "Framework/DeviceSpec.h"
#include "Framework/Logger.h"
#include "Framework/TimingInfo.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/DataTakingContext.h"
#include "Framework/RawDeviceService.h"
#include "Framework/DataSpecUtils.h"
#include "CCDB/CcdbApi.h"
#include "CommonConstants/LHCConstants.h"
#include "Framework/Signpost.h"
#include <TError.h>
#include <TMemFile.h>
#include <ctime>

O2_DECLARE_DYNAMIC_LOG(ccdb);

namespace o2::framework
{

// Fill valid routes. Here we consider only the routes which have
// Lifetime::Condition. Notice the way we do it for analysis will be
// different.
namespace
{
void fillValidRoutes(CCDBFetcherHelper& helper, std::vector<o2::framework::OutputRoute> const& outputRoutes)
{
  std::unordered_map<std::string, bool> accountedSpecs;
  for (auto& route : outputRoutes) {
    if (route.matcher.lifetime != Lifetime::Condition) {
      continue;
    }
    auto specStr = DataSpecUtils::describe(route.matcher);
    if (accountedSpecs.find(specStr) != accountedSpecs.end()) {
      continue;
    }
    accountedSpecs[specStr] = true;
    helper.routes.push_back(route);
    LOGP(info, "The following route is a condition {}", DataSpecUtils::describe(route.matcher));
    for (auto& metadata : route.matcher.metadata) {
      if (metadata.type == VariantType::String) {
        LOGP(info, "- {}: {}", metadata.name, metadata.defaultValue.asString());
      }
    }
  }
}
} // namespace

auto getOrbitResetTime(o2::pmr::vector<char> const& v) -> Long64_t
{
  Int_t previousErrorLevel = gErrorIgnoreLevel;
  gErrorIgnoreLevel = kFatal;
  TMemFile memFile("name", const_cast<char*>(v.data()), v.size(), "READ");
  gErrorIgnoreLevel = previousErrorLevel;
  if (memFile.IsZombie()) {
    throw runtime_error("CTP is Zombie");
  }
  TClass* tcl = TClass::GetClass(typeid(std::vector<Long64_t>));
  void* result = ccdb::CcdbApi::extractFromTFile(memFile, tcl);
  if (!result) {
    throw runtime_error_f("Couldn't retrieve object corresponding to %s from TFile", tcl->GetName());
  }
  memFile.Close();
  auto* ctp = (std::vector<Long64_t>*)result;
  return (*ctp)[0];
};

AlgorithmSpec CCDBHelpers::fetchFromCCDB()
{
  return adaptStateful([](CallbackService& callbacks, ConfigParamRegistry const& options, DeviceSpec const& spec) {
      std::shared_ptr<CCDBFetcherHelper> helper = std::make_shared<CCDBFetcherHelper>();
      CCDBFetcherHelper::initialiseHelper(*helper, options);
      fillValidRoutes(*helper, spec.outputs);
      /// Add a callback on stop which dumps the statistics for the caching per
      /// path
      callbacks.set<CallbackService::Id::Stop>([helper]() {
        LOGP(info, "CCDB cache miss/hit ratio:");
        for (auto& entry : helper->mapURL2UUID) {
          LOGP(info, "  {}: {}/{} ({}-{} bytes)", entry.first, entry.second.cacheMiss, entry.second.cacheHit, entry.second.minSize, entry.second.maxSize);
        }
      });

      return adaptStateless([helper](DataTakingContext& dtc, DataAllocator& allocator, TimingInfo& timingInfo) {
        auto sid = _o2_signpost_id_t{(int64_t)timingInfo.timeslice};
        O2_SIGNPOST_START(ccdb, sid, "fetchFromCCDB", "Fetching CCDB objects for timeslice %" PRIu64, (uint64_t)timingInfo.timeslice);
        static Long64_t orbitResetTime = -1;
        static size_t lastTimeUsed = -1;
        if (timingInfo.creation & DataProcessingHeader::DUMMY_CREATION_TIME_OFFSET) {
          LOGP(info, "Dummy creation time is not supported for CCDB objects. Setting creation to last one used {}.", lastTimeUsed);
          timingInfo.creation = lastTimeUsed;
        }
        lastTimeUsed = timingInfo.creation;
        // Fetch the CCDB object for the CTP
        {
          const std::string path = "CTP/Calib/OrbitReset";
          std::map<std::string, std::string> metadata;
          std::map<std::string, std::string> headers;
          std::string etag;
          bool checkValidity = std::abs(int(timingInfo.tfCounter - helper->lastCheckedTFCounterOrbReset)) >= helper->queryPeriodGlo;
          const auto url2uuid = helper->mapURL2UUID.find(path);
          if (url2uuid != helper->mapURL2UUID.end()) {
            etag = url2uuid->second.etag;
          } else {
            checkValidity = true; // never skip check if the cache is empty
          }
          O2_SIGNPOST_EVENT_EMIT(ccdb, sid, "fetchFromCCDB", "checkValidity is %{public}s for tfID %d of %{public}s",
                                 checkValidity ? "true" : "false", timingInfo.tfCounter, path.data());
          Output output{"CTP", "OrbitReset", 0};
          Long64_t newOrbitResetTime = orbitResetTime;
          auto&& v = allocator.makeVector<char>(output);
          const auto& api = helper->getAPI(path);
          if (checkValidity && (!api.isSnapshotMode() || etag.empty())) { // in the snapshot mode the object needs to be fetched only once
            helper->lastCheckedTFCounterOrbReset = timingInfo.tfCounter;
            api.loadFileToMemory(v, path, metadata, timingInfo.creation, &headers, etag, helper->createdNotAfter, helper->createdNotBefore);
            if ((headers.count("Error") != 0) || (etag.empty() && v.empty())) {
              LOGP(fatal, "Unable to find CCDB object {}/{}", path, timingInfo.creation);
              // FIXME: I should send a dummy message.
              return;
            }
            if (etag.empty()) {
              helper->mapURL2UUID[path].etag = headers["ETag"]; // update uuid
              helper->mapURL2UUID[path].cacheMiss++;
              helper->mapURL2UUID[path].minSize = std::min(v.size(), helper->mapURL2UUID[path].minSize);
              helper->mapURL2UUID[path].maxSize = std::max(v.size(), helper->mapURL2UUID[path].maxSize);
              newOrbitResetTime = getOrbitResetTime(v);
              api.appendFlatHeader(v, headers);
              auto cacheId = allocator.adoptContainer(output, std::move(v), DataAllocator::CacheStrategy::Always, header::gSerializationMethodNone);
              helper->mapURL2DPLCache[path] = cacheId;
              O2_SIGNPOST_EVENT_EMIT(ccdb, sid, "fetchFromCCDB", "Caching %{public}s for %{public}s (DPL id %" PRIu64 ")", path.data(), headers["ETag"].data(), cacheId.value);
            } else if (v.size()) { // but should be overridden by fresh object
              // somewhere here pruneFromCache should be called
              helper->mapURL2UUID[path].etag = headers["ETag"]; // update uuid
              helper->mapURL2UUID[path].cacheMiss++;
              helper->mapURL2UUID[path].minSize = std::min(v.size(), helper->mapURL2UUID[path].minSize);
              helper->mapURL2UUID[path].maxSize = std::max(v.size(), helper->mapURL2UUID[path].maxSize);
              newOrbitResetTime = getOrbitResetTime(v);
              api.appendFlatHeader(v, headers);
              auto cacheId = allocator.adoptContainer(output, std::move(v), DataAllocator::CacheStrategy::Always, header::gSerializationMethodNone);
              helper->mapURL2DPLCache[path] = cacheId;
              O2_SIGNPOST_EVENT_EMIT(ccdb, sid, "fetchFromCCDB", "Caching %{public}s for %{public}s (DPL id %" PRIu64 ")", path.data(), headers["ETag"].data(), cacheId.value);
              // one could modify the adoptContainer to take optional old cacheID to clean:
              // mapURL2DPLCache[URL] = ctx.outputs().adoptContainer(output, std::move(outputBuffer), DataAllocator::CacheStrategy::Always, mapURL2DPLCache[URL]);
            }
            // cached object is fine
          }
          auto cacheId = helper->mapURL2DPLCache[path];
          O2_SIGNPOST_EVENT_EMIT(ccdb, sid, "fetchFromCCDB", "Reusing %{public}s for %{public}s (DPL id %" PRIu64 ")", path.data(), headers["ETag"].data(), cacheId.value);
          helper->mapURL2UUID[path].cacheHit++;
          allocator.adoptFromCache(output, cacheId, header::gSerializationMethodNone);

          if (newOrbitResetTime != orbitResetTime) {
            O2_SIGNPOST_EVENT_EMIT(ccdb, sid, "fetchFromCCDB", "Orbit reset time changed from %lld to %lld", orbitResetTime, newOrbitResetTime);
            orbitResetTime = newOrbitResetTime;
            dtc.orbitResetTimeMUS = orbitResetTime;
          }
        }

        int64_t timestamp = ceil((timingInfo.firstTForbit * o2::constants::lhc::LHCOrbitNS / 1000 + orbitResetTime) / 1000); // RS ceilf precision is not enough
        if (std::abs(int64_t(timingInfo.creation) - timestamp) > helper->timeToleranceMS) {
          static bool notWarnedYet = true;
          if (notWarnedYet) {
            LOGP(warn, "timestamp {} for orbit {} and orbit reset time {} differs by >{} from the TF creation time {}, use the latter", timestamp, timingInfo.firstTForbit, orbitResetTime / 1000, helper->timeToleranceMS, timingInfo.creation);
            notWarnedYet = false;
            // apparently the orbit reset time from the CTP object makes no sense (i.e. orbit was reset for this run w/o create an object, as it happens for technical runs)
            dtc.orbitResetTimeMUS = 1000 * timingInfo.creation - timingInfo.firstTForbit * o2::constants::lhc::LHCOrbitNS / 1000;
          }
          timestamp = timingInfo.creation;
        }
        // Fetch the rest of the objects.
        O2_SIGNPOST_EVENT_EMIT(ccdb, sid, "fetchFromCCDB", "Fetching objects. Run %{public}s. OrbitResetTime %lld. Creation %lld. Timestamp %lld. firstTForbit %" PRIu32,
            dtc.runNumber.data(), orbitResetTime, timingInfo.creation, timestamp, timingInfo.firstTForbit);
        std::vector<CCDBFetcherHelper::FetchOp> ops;
        int runNumber = 0;
        std::string ccdbMetadataPrefix = "ccdb-metadata-";
        for (auto &route : helper->routes) {
          CCDBFetcherHelper::FetchOp op{.spec = route.matcher, .timestamp = timestamp, .runNumber = std::stoi(dtc.runNumber)};
          for (auto& meta : route.matcher.metadata) {
            if (meta.name == "ccdb-path") {
              op.url = meta.defaultValue.get<std::string>();
            } else if (meta.name == "ccdb-run-dependent" && meta.defaultValue.get<int>() > 0) {
              if (meta.defaultValue.get<int>() == 1) {
                op.runNumber = runNumber;
              } else if (meta.defaultValue.get<int>() == 2) {
                op.timestamp = runNumber;
              } else {
                LOGP(fatal, "Undefined ccdb-run-dependent option {} for spec {}", meta.defaultValue.get<int>(), DataSpecUtils::describe(route.matcher));
              }
            } else if (meta.name.starts_with(ccdbMetadataPrefix)) {
              std::string key = meta.name.substr(ccdbMetadataPrefix.size());
              auto value = meta.defaultValue.get<std::string>();
              O2_SIGNPOST_EVENT_EMIT(ccdb, sid, "populateCacheWith", "Adding metadata %{public}s: %{public}s to the request", key.data(), value.data());
              op.metadata.push_back({key, value});
            } else if (meta.name == "ccdb-query-rate") {
              op.queryRate = meta.defaultValue.get<int>() * helper->queryPeriodFactor;
            }
          }
          ops.push_back(op);
        }

        CCDBFetcherHelper::populateCacheWith(helper, ops, timingInfo, dtc, allocator);
        O2_SIGNPOST_END(ccdb, _o2_signpost_id_t{(int64_t)timingInfo.timeslice}, "fetchFromCCDB", "Fetching CCDB objects");
      }); });
}

} // namespace o2::framework
