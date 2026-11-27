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
#include "Framework/Plugins.h"
#include "Framework/AlgorithmSpec.h"
#include "Framework/ServiceSpec.h"
#include "Framework/ServiceMetricsInfo.h"
#include "Framework/ConfigParamDiscovery.h"
#include "Framework/Capability.h"
#include "Framework/Signpost.h"
#include "AODJAlienReaderHelpers.h"
#include "AODWriterHelpers.h"
#include "AODInputFile.h"
#include <TDataType.h>
#include <TGrid.h>
#include <fmt/format.h>
#include <algorithm>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include <unistd.h>

O2_DECLARE_DYNAMIC_LOG(analysis_support);

struct ROOTTypeInfo {
  EDataType type;
  char suffix[3];
  int size;
};

struct ROOTFileReader : o2::framework::AlgorithmPlugin {
  o2::framework::AlgorithmSpec create(o2::framework::ConfigContext const& config) override
  {
    return o2::framework::readers::AODJAlienReaderHelpers::rootFileReaderCallback(config);
  }
};

struct ROOTObjWriter : o2::framework::AlgorithmPlugin {
  o2::framework::AlgorithmSpec create(o2::framework::ConfigContext const& config) override
  {
    return o2::framework::writers::AODWriterHelpers::getOutputObjHistWriter(config);
  }
};

struct ROOTTTreeWriter : o2::framework::AlgorithmPlugin {
  o2::framework::AlgorithmSpec create(o2::framework::ConfigContext const& config) override
  {
    return o2::framework::writers::AODWriterHelpers::getOutputTTreeWriter(config);
  }
};

using namespace o2::framework;
struct RunSummary : o2::framework::ServicePlugin {
  o2::framework::ServiceSpec* create() final
  {
    return new o2::framework::ServiceSpec{
      .name = "analysis-run-summary",
      .init = [](ServiceRegistryRef ref, DeviceState&, fair::mq::ProgOptions&) -> ServiceHandle {
        return ServiceHandle{TypeIdHelpers::uniqueId<RunSummary>(), nullptr, ServiceKind::Serial, "analysis-run-summary"};
      },
      .summaryHandling = [](ServiceMetricsInfo const& info) {
        LOGP(info, "## Analysis Run Summary ##");
        /// Find the metrics of the reader and dump the list of files read.
        for (size_t mi = 0; mi < info.deviceMetricsInfos.size(); ++mi) {
          DeviceMetricsInfo &metrics = info.deviceMetricsInfos[mi];
          for (size_t li = 0; li < metrics.metricLabels.size(); ++li) {
            MetricLabel const&label = metrics.metricLabels[li];
            if (strcmp(label.label, "aod-file-read-info") != 0) {
              continue;
            }
            MetricInfo const&metric = metrics.metrics[li];
            auto &files = metrics.stringMetrics[metric.storeIdx];
            if (metric.filledMetrics) {
              LOGP(info, "### Files read stats ###");
            }
            for (size_t fi = 0; fi < metric.filledMetrics; ++fi) {
              LOGP(info, "{}", files[fi % files.size()].data);
            }
          }
          for (size_t li = 0; li < metrics.metricLabels.size(); ++li) {
            MetricLabel const&label = metrics.metricLabels[li];
            if (strcmp(label.label, "aod-file-open-info") != 0) {
              continue;
            }
            MetricInfo const&metric = metrics.metrics[li];
            auto &files = metrics.stringMetrics[metric.storeIdx];
            if (metric.filledMetrics) {
              LOGP(info, "### Files opened stats ###");
            }
            std::string lastFileRead;
            for (size_t fi = 0; fi < metric.filledMetrics; ++fi) {
              lastFileRead = files[fi % files.size()].data;
            }
            if (lastFileRead.empty() == false) {
              LOGP(info, "Last file opened: {}", lastFileRead);
            }
          }
        } },
      .kind = ServiceKind::Serial};
  }
};

std::vector<std::string> getListOfTables(AODInputFile& f)
{
  std::vector<std::string> r;
  // We should handle two cases, one where the list of tables in a  TDirectory,
  // the other one where the dataframe number is just a prefix
  std::string first = "";

  for (auto& key : f.topLevelKeys()) {
    if (!std::string_view(key.name).starts_with("DF_") && !std::string_view(key.name).starts_with("/DF_")) {
      continue;
    }
    // Objects are in a folder, list it.
    if (key.className == "TDirectoryFile" || key.className == "TDirectory") {
      r = f.keysIn(key.name);
      break;
    }

    if (key.className == "ROOT::RNTuple" || key.className == "ROOT::Experimental::RNTuple") {
      std::string const& s = key.name;
      size_t pos = s.find('-');
      // Check if '-' is found
      // Skip metaData and parentFiles
      if (pos == std::string::npos) {
        continue;
      }
      std::string t = s.substr(pos + 1);
      // If we find a duplicate table name, it means we are in the next DF and we can stop.
      if (t == first) {
        break;
      }
      if (first.empty()) {
        first = t;
      }
      // Create a new string starting after the '-'
      r.emplace_back(t);
    }
  }
  return r;
}

// Key / value pairs of the metadata, empty if there is none
auto readMetadata(AODInputFile& currentFile) -> std::map<std::string, std::string>
{
  auto metadata = currentFile.metaData();
  bool first = true;
  for (auto const& [name, value] : metadata) {
    if (first) {
      LOGP(info, "Metadata for file \"{}\":", currentFile.name());
      first = false;
    }
    LOGP(info, "- {}: {}", name, value);
  }
  return metadata;
}

auto metadataToSpecs(std::map<std::string, std::string> const& metadata) -> std::vector<ConfigParamSpec>
{
  std::vector<ConfigParamSpec> results;
  for (auto const& [name, value] : metadata) {
    results.push_back(ConfigParamSpec{"aod-metadata-" + name, VariantType::String, value, {"Metadata in AOD"}});
  }
  return results;
}

/// Opens an AO2D for metadata discovery. The factory is only needed to read
/// tables, which discovery does not do.
std::shared_ptr<AODInputFile> openForDiscovery(std::string const& filename)
{
  static RootObjectReadingFactory noTables;
  return RootAODInputFile::open(filename, noTables);
}

struct FileAtLevel {
  std::shared_ptr<AODInputFile> file; // nullptr if there is no file at this level
  std::string name;
};

// Open the first parent file of `child`
FileAtLevel openParent(FileAtLevel const& child, ConfigParamRegistry& registry)
{
  auto parentFiles = child.file->parentFiles();
  if (parentFiles.empty()) {
    LOGP(info, "No parent file found for \"{}\"", child.name);
    return {nullptr, child.name};
  }
  std::string parentFilename = parentFiles.begin()->second;
  // Do the replacement. Notice this will require changing aod-parent-base-path-replacement to be
  // a workflow option (because the metadata itself is potentially changing the topology).
  if (registry.isSet("aod-parent-base-path-replacement")) {
    auto parentFileReplacement = registry.get<std::string>("aod-parent-base-path-replacement");
    auto pos = parentFileReplacement.find(';');
    if (pos == std::string::npos) {
      throw std::runtime_error(fmt::format("Invalid syntax in aod-parent-base-path-replacement: \"{}\"", parentFileReplacement.c_str()));
    }
    auto from = parentFileReplacement.substr(0, pos);
    auto to = parentFileReplacement.substr(pos + 1);
    // Same as TString::ReplaceAll
    for (size_t at = 0; !from.empty() && (at = parentFilename.find(from, at)) != std::string::npos; at += to.size()) {
      parentFilename.replace(at, from.size(), to);
    }
  }

  if (parentFilename.starts_with("alien://") && !gGrid) {
    TGrid::Connect("alien://");
  }

  auto parentFile = openForDiscovery(parentFilename);
  if (parentFile.get() == nullptr) {
    LOGP(fatal, "Couldn't open derived file \"{}\"!", parentFilename);
  }
  return {std::move(parentFile), parentFilename};
}

// What was found at a given parent level
struct LevelMetadata {
  bool found = false; // the file at this level exists
  std::string source;
  std::map<std::string, std::string> metadata;
  std::vector<std::string> tables;
};

struct DiscoverMetadataInAOD : o2::framework::ConfigDiscoveryPlugin {
  ConfigDiscovery* create() override
  {
    return new ConfigDiscovery{
      .init = []() {},
      .discover = [](ConfigParamRegistry& registry, int argc, char** argv) -> std::vector<ConfigParamSpec> {
        auto filename = registry.get<std::string>("aod-file");
        if (filename.empty()) {
          return {};
        }
        if (filename.at(0) == '@') {
          filename.erase(0, 1);
          // read the text file and set filename to the contents of the first line
          std::ifstream file(filename);
          if (!file.is_open()) {
            LOGP(fatal, "Couldn't open file \"{}\"!", filename);
          }
          std::getline(file, filename);
          file.close();
        }
        if (filename.rfind("alien://", 0) == 0 && !gGrid) {
          TGrid::Connect("alien://");
        }
        LOGP(info, "Loading metadata from file {} in PID {}", filename, getpid());
        auto currentFile = openForDiscovery(filename);
        if (currentFile.get() == nullptr) {
          throw std::runtime_error(fmt::format("Couldn't open file \"{}\"!", filename));
        }

        // The files opened so far, by level. Ends with an empty file if the chain of parents is exhausted.
        std::vector<FileAtLevel> chain{{currentFile, filename}};
        // Each level is read only once, even if several origins map to it
        std::map<int, LevelMetadata> levels;
        auto readLevel = [&](int level) -> LevelMetadata const& {
          if (auto it = levels.find(level); it != levels.end()) {
            return it->second;
          }
          while (chain.back().file && chain.size() <= static_cast<size_t>(level)) {
            chain.push_back(openParent(chain.back(), registry));
          }
          LevelMetadata data;
          auto const& atLevel = chain[std::min(static_cast<size_t>(level), chain.size() - 1)];
          if (atLevel.file) {
            data.found = true;
            data.source = atLevel.name;
            data.metadata = readMetadata(*atLevel.file);
            data.tables = getListOfTables(*atLevel.file);
          }
          return levels.emplace(level, std::move(data)).first->second;
        };
        int parentAccessLevel = 0;
        if (registry.isSet("aod-parent-access-level")) {
          parentAccessLevel = registry.get<int>("aod-parent-access-level");
        }

        // The plain aod-metadata-* options, from the file itself or from the closest parent which has them
        LevelMetadata const* plain = nullptr;
        auto const& top = readLevel(0);
        std::vector<ConfigParamSpec> results = metadataToSpecs(top.metadata);
        if (!top.tables.empty()) {
          results.push_back(ConfigParamSpec{"aod-metadata-tables", VariantType::ArrayString, top.tables, {"Tables in first AOD"}});
        }
        if (!top.metadata.empty()) {
          // Found metadata already in the main file.
          results.push_back(ConfigParamSpec{"aod-metadata-source", VariantType::String, filename, {"File from which the metadata was extracted."}});
          plain = &top;
        } else if (parentAccessLevel == 0) {
          LOGP(info, "No metadata found in file \"{}\" and parent level 0 prevents further lookup.", filename);
          results.push_back(ConfigParamSpec{"aod-metadata-disable", VariantType::String, "1", {"Metadata not found in AOD"}});
        } else {
          // Lets try in the parent files, up to the allowed parent level.
          LOGP(info, "No metadata found in file \"{}\", checking in its parents.", filename);
          for (int level = 1; level <= parentAccessLevel && plain == nullptr; ++level) {
            auto const& parent = readLevel(level);
            if (!parent.found) {
              break;
            }
            results = metadataToSpecs(parent.metadata);
            if (parent.metadata.empty()) {
              LOGP(info, "No metadata found in parent file \"{}\" at level {}", parent.source, level);
              continue;
            }
            if (!parent.tables.empty()) {
              results.push_back(ConfigParamSpec{"aod-metadata-tables", VariantType::ArrayString, parent.tables, {"Tables in first AOD"}});
            }
            results.push_back(ConfigParamSpec{"aod-metadata-source", VariantType::String, filename, {"File from which the metadata was extracted."}});
            plain = &parent;
          }
          if (plain == nullptr) {
            LOGP(info, "No metadata found in file \"{}\" nor in its parents", filename);
            results.push_back(ConfigParamSpec{"aod-metadata-disable", VariantType::String, "1", {"Metadata not found in AOD"}});
          }
        }

        if (!registry.hasOption("aod-origin-level-map")) {
          return results;
        }

        // The metadata of the level each origin reads, one row per origin. The columns are
        // all the keys found, including the plain ones, so that an empty cell always means
        // that the key is missing for that origin.
        auto levelMap = registry.get<LabeledArray<int>>("aod-origin-level-map");
        auto origins = levelMap.getLabelsRows();
        std::vector<std::string> keys;
        auto addKeys = [&keys](std::map<std::string, std::string> const& metadata) {
          for (auto const& [key, value] : metadata) {
            if (std::ranges::find(keys, key) == keys.end()) {
              keys.push_back(key);
            }
          }
        };
        if (plain != nullptr) {
          addKeys(plain->metadata);
        }
        for (auto i = 0U; i < levelMap.rows(); ++i) {
          auto level = levelMap.get(i, 0U);
          if (level > parentAccessLevel) {
            LOGP(fatal, "Origin {} is mapped to parent level {}, but aod-parent-access-level is {}", origins[i], level, parentAccessLevel);
          }
          addKeys(readLevel(level).metadata);
        }
        auto sourceColumn = keys.size();
        keys.emplace_back("source");
        auto disableColumn = keys.size();
        keys.emplace_back("disable");

        std::vector<std::string> values(origins.size() * keys.size());
        for (auto i = 0U; i < origins.size(); ++i) {
          auto const& data = readLevel(levelMap.get(i, 0U));
          auto* row = values.data() + i * keys.size();
          for (auto const& [key, value] : data.metadata) {
            row[std::ranges::find(keys, key) - keys.begin()] = value;
          }
          row[sourceColumn] = data.source;
          if (data.metadata.empty()) {
            LOGP(info, "Metadata info missing for origin {}", origins[i]);
            row[disableColumn] = "1";
          }
        }
        LabeledArray<std::string> originMetadata{values.data(), static_cast<uint32_t>(origins.size()), static_cast<uint32_t>(keys.size()), origins, keys};
        results.push_back(ConfigParamSpec{"aod-metadata-origins", VariantType::LabeledArrayString, originMetadata, {"Metadata in AOD for each origin: one row per origin, one column per key"}});
        for (auto i = 0U; i < origins.size(); ++i) {
          auto const& tables = readLevel(levelMap.get(i, 0U)).tables;
          if (!tables.empty()) {
            results.push_back(ConfigParamSpec{"aod-metadata-origin-" + origins[i] + "-tables", VariantType::ArrayString, tables, {"Tables in the AOD read for this origin"}});
          }
        }
        return results;
      }};
  }
};

DEFINE_DPL_PLUGINS_BEGIN
DEFINE_DPL_PLUGIN_INSTANCE(ROOTFileReader, CustomAlgorithm);
DEFINE_DPL_PLUGIN_INSTANCE(ROOTObjWriter, CustomAlgorithm);
DEFINE_DPL_PLUGIN_INSTANCE(ROOTTTreeWriter, CustomAlgorithm);
DEFINE_DPL_PLUGIN_INSTANCE(RunSummary, CustomService);
DEFINE_DPL_PLUGIN_INSTANCE(DiscoverMetadataInAOD, ConfigDiscovery);
DEFINE_DPL_PLUGINS_END
