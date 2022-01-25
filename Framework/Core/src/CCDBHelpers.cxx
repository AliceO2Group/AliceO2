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

#include "CCDBHelpers.h"
#include "Framework/DeviceSpec.h"
#include "Framework/Logger.h"
#include "Framework/TimingInfo.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/DataTakingContext.h"
#include "Framework/RawDeviceService.h"
#include "CCDB/CcdbApi.h"
#include "CommonConstants/LHCConstants.h"
#include <typeinfo>
#include <TError.h>
#include <TMemFile.h>

namespace o2::framework
{

struct CCDBFetcherHelper {
  struct CCDBCacheInfo {
    std::string path;
    int64_t cacheId;
    std::string etag;
  };

  struct RemapMatcher {
    std::string path;
  };

  struct RemapTarget {
    std::string url;
  };

  std::unordered_map<std::string, std::string> mapURL2UUID;
  std::unordered_map<std::string, DataAllocator::CacheId> mapURL2DPLCache;
  std::string createdNotBefore = "0";
  std::string createdNotAfter = "3385078236000";

  o2::ccdb::CcdbApi api;
  std::vector<OutputRoute> routes;
  std::map<int64_t, CCDBCacheInfo> cache;
  std::unordered_map<std::string, std::string> remappings;
};

bool isPrefix(std::string_view prefix, std::string_view full)
{
  return prefix == full.substr(0, prefix.size());
}

CCDBHelpers::ParserResult CCDBHelpers::parseRemappings(char const* str)
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
        if (*str != '/' && (strncmp("http://", str, 6) != 0) && (strncmp("https://", str, 7) != 0)) {
          return {remappings, "URL should start with either / or http:// / https://"};
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

AlgorithmSpec CCDBHelpers::fetchFromCCDB()
{
  return adaptStateful([](ConfigParamRegistry const& options, DeviceSpec const& spec) { 
      std::shared_ptr<CCDBFetcherHelper> helper = std::make_shared<CCDBFetcherHelper>();
      auto backend = options.get<std::string>("condition-backend");
      LOGP(info, "CCDB Backend at: {}", backend);
      helper->api.init(options.get<std::string>("condition-backend"));
      helper->createdNotBefore = std::to_string(options.get<int64_t>("condition-not-before"));
      helper->createdNotAfter = std::to_string(options.get<int64_t>("condition-not-after"));
      auto remapString = options.get<std::string>("condition-remap");
      ParserResult result = CCDBHelpers::parseRemappings(remapString.c_str());
      if (!result.error.empty()) {
        throw runtime_error_f("Error while parsing remapping string %s", result.error.c_str());
      }

      helper->remappings = result.remappings;

      for (auto &route : spec.outputs) {
        if (route.matcher.lifetime != Lifetime::Condition) {
          continue;
        }
        helper->routes.push_back(route);
        LOGP(info, "The following route is a condition {}", route.matcher);
        for (auto& metadata : route.matcher.metadata) {
          if (metadata.type == VariantType::String) { 
            LOGP(info, "- {}: {}", metadata.name, metadata.defaultValue);
          }
        }
      }

      auto getOrbitResetTime = [](o2::pmr::vector<char> const& v) -> Long64_t {
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
        std::vector<Long64_t>* ctp = (std::vector<Long64_t>*)result;
        return (*ctp)[0];
      };

      return adaptStateless([helper, &getOrbitResetTime](DataTakingContext& dtc, DataAllocator& allocator, TimingInfo& timingInfo) {
        static Long64_t orbitResetTime = -1;
        // Fetch the CCDB object for the CTP
        {
          // FIXME: this (the static) is needed because for now I cannot get
          // a pointer for the cachedObject in the fetcher itself.
          // Will be fixed at a later point.
          std::string path = "CTP/Calib/OrbitReset";
          std::map<std::string, std::string> metadata;
          std::map<std::string, std::string> headers;
          std::string etag;
          const auto url2uuid = helper->mapURL2UUID.find(path);
          if (url2uuid != helper->mapURL2UUID.end()) {
            etag = url2uuid->second;
          }
          Output output{"CTP", "OrbitReset", 0, Lifetime::Condition};
          auto&& v = allocator.makeVector<char>(output);
          helper->api.loadFileToMemory(v, path, metadata, timingInfo.creation,
                                       &headers, etag,
                                       helper->createdNotAfter,
                                       helper->createdNotBefore);

          if ((headers.count("Error") != 0) || (etag.empty() && v.empty())) {
            LOGP(error, "Unable to find object {}/{}", path, timingInfo.creation);
            //FIXME: I should send a dummy message.
            return;
          }
          Long64_t newOrbitResetTime = orbitResetTime;
          if (etag.empty()) {
            helper->mapURL2UUID[path] = headers["ETag"]; // update uuid
            newOrbitResetTime = getOrbitResetTime(v);
            auto cacheId = allocator.adoptContainer(output, std::move(v), true, header::gSerializationMethodNone);
            helper->mapURL2DPLCache[path] = cacheId;
            LOGP(debug, "Caching {} for {} (DPL id {})", path, headers["ETag"], cacheId.value);
          } else if (v.size()) { // but should be overridden by fresh object
            // somewhere here pruneFromCache should be called
            helper->mapURL2UUID[path] = headers["ETag"]; // update uuid
            newOrbitResetTime = getOrbitResetTime(v);
            auto cacheId = allocator.adoptContainer(output, std::move(v), true, header::gSerializationMethodNone);
            helper->mapURL2DPLCache[path] = cacheId;
            LOGP(debug, "Caching {} for {} (DPL id {})", path, headers["ETag"], cacheId.value);
            // one could modify the    adoptContainer to take optional old cacheID to clean:
            // mapURL2DPLCache[URL] = ctx.outputs().adoptContainer(output, std::move(outputBuffer), true, mapURL2DPLCache[URL]);
          } else { // cached object is fine
            auto cacheId = helper->mapURL2DPLCache[path];
            LOGP(debug, "Reusing {} for {}", cacheId.value, path);
            allocator.adoptFromCache(output, cacheId, header::gSerializationMethodNone);
            // We need to find a way to get "v" also in this case.
            // orbitResetTime = getOrbitResetTime(v);
            // the outputBuffer was not used, can we destroy it?
          }
          if (newOrbitResetTime != orbitResetTime) {
            LOGP(info, "Orbit reset time now at {} (was {})",
                 newOrbitResetTime, orbitResetTime);
            orbitResetTime = newOrbitResetTime;
          }
        }

        int64_t timestamp = ceilf((timingInfo.firstTFOrbit * o2::constants::lhc::LHCOrbitNS / 1000 + orbitResetTime) / 1000);
        // Fetch the rest of the objects.
        LOGP(info, "Fetching objects. Run: {}. OrbitResetTime: {}, Timestamp: {}, firstTFOrbit: {}",
             dtc.runNumber, dtc.orbitResetTime, timestamp, timingInfo.firstTFOrbit);
        std::string ccdbMetadataPrefix = "ccdb-metadata-";

        for (auto& route : helper->routes) {
          LOGP(info, "Fetching object for route {}", route.matcher);

          auto concrete = DataSpecUtils::asConcreteDataMatcher(route.matcher);
          Output output{concrete.origin, concrete.description, concrete.subSpec, route.matcher.lifetime};
          auto&& v = allocator.makeVector<char>(output);
          std::map<std::string, std::string> metadata;
          std::map<std::string, std::string> headers;
          std::string path = "";
          std::string etag = "";
          for (auto& meta : route.matcher.metadata) {
            if (meta.name == "ccdb-path") {
              path = meta.defaultValue.get<std::string>();
              auto prefix = helper->remappings.find(path);
              // FIXME: for now assume that we can pass the whole URL to
              // the CCDB API. It might be better to have multiple instances...
              if (prefix != helper->remappings.end()) {
                LOGP(error, "Remapping {} to {}", path, prefix->second + path);
                path = prefix->second + path;
              }
            } else if (isPrefix(ccdbMetadataPrefix, meta.name)) {
              std::string key = meta.name.substr(ccdbMetadataPrefix.size());
              auto value = meta.defaultValue.get<std::string>();
              LOGP(debug, "Adding metadata {}: {} to the request", key, value);
              metadata[key] = value;
            }
          }
          const auto url2uuid = helper->mapURL2UUID.find(path);
          if (url2uuid != helper->mapURL2UUID.end()) {
            etag = url2uuid->second;
          }

          helper->api.loadFileToMemory(v, path, metadata, timestamp, &headers, etag, helper->createdNotAfter, helper->createdNotBefore);
          if ((headers.count("Error") != 0) || (etag.empty() && v.empty())) {
            LOGP(debug, "Unable to find object {}/{}", path, timingInfo.timeslice);
            //FIXME: I should send a dummy message.
            continue;
          }
          if (etag.empty()) {
            helper->mapURL2UUID[path] = headers["ETag"]; // update uuid
            auto cacheId = allocator.adoptContainer(output, std::move(v), true, header::gSerializationMethodCCDB);
            helper->mapURL2DPLCache[path] = cacheId;
            LOGP(debug, "Caching {} for {} (DPL id {})", path, headers["ETag"], cacheId.value);
            continue;
          }
          if (v.size()) { // but should be overridden by fresh object
            // somewhere here pruneFromCache should be called
            helper->mapURL2UUID[path] = headers["ETag"]; // update uuid
            auto cacheId = allocator.adoptContainer(output, std::move(v), true, header::gSerializationMethodCCDB);
            helper->mapURL2DPLCache[path] = cacheId;
            LOGP(info, "Caching {} for {} (DPL id {})", path, headers["ETag"], cacheId.value);
            // one could modify the    adoptContainer to take optional old cacheID to clean:
            // mapURL2DPLCache[URL] = ctx.outputs().adoptContainer(output, std::move(outputBuffer), true, mapURL2DPLCache[URL]);
          } else { // cached object is fine
            auto cacheId = helper->mapURL2DPLCache[path];
            LOGP(info, "Reusing {} for {}", cacheId.value, path);
            allocator.adoptFromCache(output, cacheId, header::gSerializationMethodCCDB);
            // the outputBuffer was not used, can we destroy it?
          }
        }
      }); });
}

} // namespace o2::framework
