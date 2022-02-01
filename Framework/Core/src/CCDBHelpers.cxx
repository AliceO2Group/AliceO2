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
#include "CommonDataFormat/InteractionRecord.h"
#include "Framework/AnalysisDataModel.h"
#include "Framework/AnalysisHelpers.h"
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
  void* context = nullptr; // type erased pointer to the context for analysis or reco
};

/// Holds information about the analysis
struct CCDBAnalysisContext {
  std::map<int, int>* mapStartOrbit = nullptr; /// Map of the starting orbit for the run
  std::map<int, long> mapRunToTimestamp;       /// Cache of processed run numbers
  int lastRunNumber = 0;                       /// Last run number processed
  long runNumberTimeStamp = 0;                 /// Timestamp of the run number, used in the process function to work out the timestamp of the BC
  uint32_t initialOrbit = 0;                   /// Index of the first orbit of the run number, used in the process function to evaluate the offset with respect to the starting of the run
  static constexpr uint16_t initialBC = 0;     /// Index of the initial bc, exact bc number not relevant due to ms precision of timestamps
  InteractionRecord initialIR;                 /// Initial interaction record, used to compute the delta with respect to the start of the run
  bool isRun2MC;
  Produces<aod::Timestamps> timestampTable; /// Table with SOR timestamps produced by the task
};

struct CCDBRecoAnalysisContext {
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

auto createBackend(ConfigParamRegistry const& options, DeviceSpec const& spec) -> std::shared_ptr<CCDBFetcherHelper>
{
  std::shared_ptr<CCDBFetcherHelper> helper = std::make_shared<CCDBFetcherHelper>();
  auto backend = options.get<std::string>("condition-backend");
  LOGP(info, "CCDB Backend at: {}", backend);
  helper->api.init(options.get<std::string>("condition-backend"));
  helper->createdNotBefore = std::to_string(options.get<int64_t>("condition-not-before"));
  helper->createdNotAfter = std::to_string(options.get<int64_t>("condition-not-after"));
  auto remapString = options.get<std::string>("condition-remap");
  CCDBHelpers::ParserResult result = CCDBHelpers::parseRemappings(remapString.c_str());
  if (!result.error.empty()) {
    throw runtime_error_f("Error while parsing remapping string %s", result.error.c_str());
  }

  helper->remappings = result.remappings;

  for (auto& route : spec.outputs) {
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
  return helper;
}

auto populateCacheWith(std::shared_ptr<CCDBFetcherHelper> const& helper,
                       int64_t timestamp,
                       TimingInfo& timingInfo,
                       DataAllocator& allocator,
                       std::string runNumber) -> void
{
  // For Giulio: the dtc.orbitResetTime is wrong, it is assigned from the dph->creation, why?
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
      } else if (meta.name == "ccdb-run-dependent") {
        metadata["runNumber"] = runNumber;
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
      // FIXME: I should send a dummy message.
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
}

auto recoCCDBCallback(std::shared_ptr<CCDBFetcherHelper> helper)
{
  return adaptStateless([helper](DataTakingContext& dtc,
                                 DataAllocator& allocator,
                                 TimingInfo& timingInfo,
                                 DataTakingContext& dataTakingContext) {
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
        // FIXME: I should send a dummy message.
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

    int64_t timestamp = ceil((timingInfo.firstTFOrbit * o2::constants::lhc::LHCOrbitNS / 1000 + orbitResetTime) / 1000); // RS ceilf precision is not enough
    // Fetch the rest of the objects.
    LOGP(info, "Fetching objects. Run: {}. OrbitResetTime: {}, Creation: {}, Timestamp: {}, firstTFOrbit: {}",
         dtc.runNumber, orbitResetTime, timingInfo.creation, timestamp, timingInfo.firstTFOrbit);

    populateCacheWith(helper, timestamp, timingInfo, allocator, dataTakingContext.runNumber);
  });
}

void* decodeObject(char* buffer, size_t s, std::type_info const& tinfo)
{
  void* result = nullptr;
  Int_t previousErrorLevel = gErrorIgnoreLevel;
  gErrorIgnoreLevel = kFatal;
  TMemFile memFile("name", buffer, s, "READ");
  gErrorIgnoreLevel = previousErrorLevel;
  if (memFile.IsZombie()) {
    return nullptr;
  }
  TClass* tcl = TClass::GetClass(tinfo);
  result = ccdb::CcdbApi::extractFromTFile(memFile, tcl);
  if (!result) {
    throw runtime_error_f("Couldn't retrieve object corresponding to %s from TFile", tcl->GetName());
  }
  memFile.Close();
  return result;
}

void initAnalysis(std::shared_ptr<CCDBFetcherHelper> helper)
{
  std::map<std::string, std::string> metadata;
  std::map<std::string, std::string> headers;
  std::string etag;
  std::string path = "GRP/StartOrbiot";
  auto now = std::chrono::steady_clock::now();
  uint64_t timestamp = std::chrono::duration<double, std::milli>(now.time_since_epoch()).count();
  o2::pmr::vector<char> v;
  helper->api.loadFileToMemory(v, path, metadata, timestamp,
                               &headers, etag,
                               helper->createdNotAfter,
                               helper->createdNotBefore);
  auto* context = (CCDBAnalysisContext*)helper->context;
  context->mapStartOrbit = (std::map<int, int>*)decodeObject(v.data(), v.size(), typeid(std::map<int, int>));
  if (!context->mapStartOrbit) {
    LOGP(fatal, "Cannot find map of SOR orbits in CCDB in path {}", path);
  }
}

void makeInitialOrbit(aod::BC const& bunchCrossing, CCDBAnalysisContext& context)
{
  if (!context.mapStartOrbit->count(bunchCrossing.runNumber())) {
    LOGP(fatal, "Cannot find run {} in mapStartOrbit map", bunchCrossing.runNumber());
  }
  context.initialOrbit = context.mapStartOrbit->at(bunchCrossing.runNumber());
  context.initialIR.bc = context.initialBC;
  context.initialIR.orbit = context.initialOrbit;
  // Setting lastCall
  LOGF(debug, "Setting the last call of the timestamp for run %i to %llu", bunchCrossing.runNumber(), context.runNumberTimeStamp);
  context.lastRunNumber = bunchCrossing.runNumber(); // Setting latest run number information
}

void createTimestampTable(aod::BC const& bc, o2::ccdb::CcdbApi& ccdb, CCDBAnalysisContext& context, std::string const& rct_path, bool verbose)
{
  // First: we need to set the timestamp from the run number.
  // This is done with caching if the run number of the BC was already processed before
  // If not the timestamp of the run number from BC is queried from CCDB and added to the cache
  if (bc.runNumber() == context.lastRunNumber) { // The run number coincides to the last run processed
    LOGF(debug, "Using timestamp from last call");
  } else if (context.mapRunToTimestamp.count(bc.runNumber())) { // The run number was already requested before: getting it from cache!
    LOGF(debug, "Getting timestamp from cache");
    context.runNumberTimeStamp = context.mapRunToTimestamp[bc.runNumber()];
    makeInitialOrbit(bc, context);
  } else { // The run was not requested before: need to acccess CCDB!
    LOGF(debug, "Getting timestamp from CCDB");
    std::map<std::string, std::string> metadata, headers;
    const std::string run_path = fmt::format("{}/{}", rct_path, bc.runNumber());
    headers = ccdb.retrieveHeaders(run_path, metadata, -1);
    if (headers.count("SOR") == 0) {
      LOGF(fatal, "Cannot find run-number to timestamp in path '%s'.", run_path.data());
    }
    context.runNumberTimeStamp = atol(headers["SOR"].c_str()); // timestamp of the SOR in ms

    // Adding the timestamp to the cache map
    std::pair<std::map<int, long>::iterator, bool> check;
    check = context.mapRunToTimestamp.insert(std::pair<int, long>(bc.runNumber(), context.runNumberTimeStamp));
    if (!check.second) {
      LOGF(fatal, "Run number %i already existed with a timestamp of %llu", bc.runNumber(), check.first->second);
    }
    makeInitialOrbit(bc, context);
    LOGF(info, "Add new run number %i with timestamp %llu to cache", bc.runNumber(), context.runNumberTimeStamp);
  }

  if (verbose) {
    LOGF(info, "Run-number to timestamp found! %i %llu ms", bc.runNumber(), context.runNumberTimeStamp);
  }
  const uint16_t currentBC = context.isRun2MC ? context.initialBC : (bc.globalBC() % o2::constants::lhc::LHCMaxBunches);
  const uint32_t currentOrbit = context.isRun2MC ? context.initialOrbit : (bc.globalBC() / o2::constants::lhc::LHCMaxBunches);
  const InteractionRecord currentIR(currentBC, currentOrbit);
  context.timestampTable(context.runNumberTimeStamp + (currentIR - context.initialIR).bc2ns() * 1e-6);
}

auto analysisCCDBCallback(std::shared_ptr<CCDBFetcherHelper> helper)
{
  return adaptStateless([helper](InputRecord& record,
                                 DataAllocator& allocator) {
    auto t = record.get<TableConsumer>(aod::MetadataTrait<aod::BCs>::metadata::tableLabel());
    auto bcs = aod::BCs{t->asArrowTable()};
    auto* context = (CCDBAnalysisContext*)helper->context;
    context->timestampTable.resetCursor(allocator.make<TableBuilder>(context->timestampTable.ref()));
    for (auto& bc : bcs) {
      createTimestampTable(bc, helper->api, *context, "RCT/RunInformation", true);
      // populateCacheWith(helper, timestamp, timingInfo, allocator, dataTakingContext.runNumber);
    }
  });
}

/// * for a given timeframe in analysis, I need to find the
/// table which provides me the BC
/// * for all the BCs in that table, I need to find the associated
///   ccdb objects for each of them. Possibly with a unique ptr.
AlgorithmSpec CCDBHelpers::fetchFromCCDB(Mode ccdbMode)
{
  return adaptStateful([ccdbMode](ConfigParamRegistry const& options, DeviceSpec const& spec) {
    std::shared_ptr<CCDBFetcherHelper> helper = createBackend(options, spec);

    switch (ccdbMode) {
      case Mode::Analysis: {
        auto* context = new CCDBAnalysisContext();
        initAnalysis(helper);
        helper->context = context;
        return analysisCCDBCallback(helper);
      }
      case Mode::Data:
      case Mode::MC:
      case Mode::Run2:
        return recoCCDBCallback(helper);
    }
  });
}

} // namespace o2::framework
