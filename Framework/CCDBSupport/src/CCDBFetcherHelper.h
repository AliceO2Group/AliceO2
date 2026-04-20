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
#ifndef O2_FRAMEWORK_CCDBFETCHERHELPER_H_
#define O2_FRAMEWORK_CCDBFETCHERHELPER_H_

#include "Framework/OutputRoute.h"
#include "Framework/DataAllocator.h"
#include "CCDB/CcdbApi.h"
#include <unordered_map>
#include <string>

namespace o2::framework
{

struct DataTakingContext;

struct CCDBFetcherHelper {
  struct CCDBCacheInfo {
    std::string etag;
    size_t cacheValidUntil = 0;
    size_t cachePopulatedAt = 0;
    size_t cacheMiss = 0;
    size_t cacheHit = 0;
    size_t size = 0L;
    size_t minSize = -1ULL;
    size_t maxSize = 0;
    int lastCheckedTF = 0;
    int lastCheckedSlice = 0;
  };

  struct RemapMatcher {
    std::string path;
  };

  struct RemapTarget {
    std::string url;
  };

  struct ParserResult {
    std::unordered_map<std::string, std::string> remappings;
    std::string error;
  };

  struct MetadataEntry {
    std::string key;
    std::string value;
  };

  // A fetch operation.
  struct FetchOp {
    // Where to put the blob
    OutputSpec& spec;
    // The url to fetch
    std::string url = "";
    // The timestamp to use
    int64_t timestamp = 0;
    // The run to use
    int runNumber = 0;
    // Wether or not the thing is run dependent
    int runDependent = 0;
    // Actual metadata
    std::vector<MetadataEntry> metadata = {};
    // Query rate
    int queryRate = 0;
  };

  // Where the data has been fetched
  struct Response {
    // CacheId / Pointer to the actual data
    DataAllocator::CacheId id;
    // The size of the buffer
    size_t size = 0;
    // Where to actually
    FetchOp* request = nullptr;
  };

  static ParserResult parseRemappings(char const*);

  size_t totalFetchedBytes = 0;
  size_t totalRequestedBytes = 0;
  std::unordered_map<std::string, CCDBCacheInfo> mapURL2UUID;
  std::unordered_map<std::string, DataAllocator::CacheId> mapURL2DPLCache;
  std::string createdNotBefore = "0";
  std::string createdNotAfter = "3385078236000";
  std::unordered_map<std::string, o2::ccdb::CcdbApi> apis;
  std::vector<OutputRoute> routes;
  std::unordered_map<std::string, std::string> remappings;
  uint32_t lastCheckedTFCounterOrbReset = 0; // last checkecked TFcounter for bulk check
  int queryPeriodGlo = 1;
  int queryPeriodFactor = 1;
  int64_t timeToleranceMS = 5000;
  int useTFSlice = 0; // if non-zero, use TFslice instead of TFcounter for the validity check. If > requested checking rate, add additional check on |lastTFchecked - TCcounter|<=useTFSlice

  o2::ccdb::CcdbApi& getAPI(const std::string& path);
  static void initialiseHelper(CCDBFetcherHelper& helper, ConfigParamRegistry const& options);
  static auto populateCacheWith(std::shared_ptr<CCDBFetcherHelper> const& helper,
                                std::vector<FetchOp> const& ops,
                                TimingInfo& timingInfo,
                                DataTakingContext& dtc,
                                DataAllocator& allocator) -> std::vector<Response>;
};

} // namespace o2::framework

#endif // O2_FRAMEWORK_CCDBFETCHERHELPER_H_
