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

namespace o2::framework
{

struct CCDBFetcherHelper {
  struct CCDBCacheInfo {
    std::string path;
    int64_t cacheId;
    std::string etag;
  };

  std::unordered_map<std::string, std::string> mapURL2UUID;
  std::unordered_map<std::string, DataAllocator::CacheId> mapURL2DPLCache;

  o2::ccdb::CcdbApi api;
  std::vector<OutputRoute> routes;
  std::map<int64_t, CCDBCacheInfo> cache;
};

AlgorithmSpec CCDBHelpers::fetchFromCCDB()
{
  return adaptStateful([](ConfigParamRegistry const& options, DeviceSpec const& spec) { 
      std::shared_ptr<CCDBFetcherHelper> helper = std::make_shared<CCDBFetcherHelper>();
      auto backend = options.get<std::string>("condition-backend");
      LOGP(info, "CCDB Backend at: {}", backend);
      helper->api.init(options.get<std::string>("condition-backend"));

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
      
      return adaptStateless([helper](RawDeviceService &device, DataTakingContext& dtc, DataAllocator& allocator, TimingInfo &timingInfo) {
          LOGP(info, "Fetching objects. Run: {}. OrbitResetTime: {}", dtc.runNumber, dtc.orbitResetTime);

          for (auto &route : helper->routes) {
            LOGP(info, "Fetching object for route {}", route.matcher);

            auto concrete = DataSpecUtils::asConcreteDataMatcher(route.matcher);
            Output output{concrete.origin, concrete.description, concrete.subSpec, route.matcher.lifetime};
            auto &&v = allocator.makeVector<char>(output);
            std::map<std::string, std::string> metadata;
            std::map<std::string, std::string> headers;
            std::string path = "";
            std::string etag = "";
            std::string createdNotBefore = "0";
            std::string createdNotAfter = "0";
            for (auto& meta : route.matcher.metadata) {
              if (meta.name == "ccdb-path") {
                path = meta.defaultValue.get<std::string>();
                break;
              }
            }
            const auto url2uuid = helper->mapURL2UUID.find(path);
            if (url2uuid != helper->mapURL2UUID.end()) {
              etag = url2uuid->second;
            }

            LOGP(info, "{} {} {}\n", timingInfo.timeslice, timingInfo.tfCounter, timingInfo.creation);
            helper->api.loadFileToMemory(v, path, metadata, timingInfo.timeslice, &headers, etag, createdNotAfter, createdNotBefore);
            if ((headers.count("Error") != 0) || (etag.empty() && v.empty())) {
              LOGP(error, "Unable to find object {}/{}", path, timingInfo.timeslice);
              //FIXME: I should send a dummy message.
              continue;
            }
            if (etag.empty()) {
              helper->mapURL2UUID[path] = headers["ETag"]; // update uuid
              auto cacheId = allocator.adoptContainer(output, std::move(v), true, header::gSerializationMethodCCDB);
              helper->mapURL2DPLCache[path] = cacheId;
              LOGP(info, "Caching {} for {} (DPL id {})", path, headers["ETag"], cacheId.value);
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
          device.waitFor(5000);
                              }); });
}

} // namespace o2::framework
