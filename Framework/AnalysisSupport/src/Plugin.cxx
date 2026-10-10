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
#include <TFile.h>
#include <TMap.h>
#include <TGrid.h>
#include <TObjString.h>
#include <TString.h>
#include <fmt/format.h>
#include <algorithm>
#include <map>
#include <memory>

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

std::vector<std::string> getListOfTables(TFile* f)
{
  std::vector<std::string> r;
  TList* keyList = f->GetListOfKeys();
  // We should handle two cases, one where the list of tables in a  TDirectory,
  // the other one where the dataframe number is just a prefix
  std::string first = "";

  for (auto key : *keyList) {
    if (!std::string_view(key->GetName()).starts_with("DF_") && !std::string_view(key->GetName()).starts_with("/DF_")) {
      continue;
    }
    auto* d = (TDirectory*)f->GetObjectChecked(key->GetName(), TClass::GetClass("TDirectory"));
    // Objects are in a folder, list it.
    if (d) {
      TList* branchList = d->GetListOfKeys();
      for (auto b : *branchList) {
        r.emplace_back(b->GetName());
      }
      break;
    }

#if __has_include(<ROOT/RFieldBase.hxx>)
    void* v = f->GetObjectChecked(key->GetName(), TClass::GetClass("ROOT::RNTuple"));
#else
    void* v = f->GetObjectChecked(key->GetName(), TClass::GetClass("ROOT::Experimental::RNTuple"));
#endif
    if (v) {
      std::string s = key->GetName();
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

// Ordered key / value pairs of the metadata, empty if there is none
auto readMetadata(TFile* currentFile) -> std::vector<std::pair<std::string, std::string>>
{
  // Get the metadata, if any
  auto m = (TMap*)currentFile->Get("metaData");
  if (!m) {
    return {};
  }
  std::vector<std::pair<std::string, std::string>> results;
  auto it = m->MakeIterator();

  bool first = true;
  while (auto obj = it->Next()) {
    if (first) {
      LOGP(info, "Metadata for file \"{}\":", currentFile->GetName());
      first = false;
    }
    auto objString = (TObjString*)m->GetValue(obj);
    LOGP(info, "- {}: {}", obj->GetName(), objString->String().Data());
    results.emplace_back(obj->GetName(), objString->String().Data());
  }

  return results;
}

auto metadataToSpecs(std::vector<std::pair<std::string, std::string>> const& metadata) -> std::vector<ConfigParamSpec>
{
  std::vector<ConfigParamSpec> results;
  for (auto const& [key, value] : metadata) {
    results.push_back(ConfigParamSpec{"aod-metadata-" + key, VariantType::String, value, {"Metadata in AOD"}});
  }
  return results;
}

auto metadataKeys(std::vector<std::pair<std::string, std::string>> const& metadata) -> std::vector<std::string>
{
  std::vector<std::string> keys;
  for (auto const& [key, value] : metadata) {
    keys.push_back(key);
  }
  return keys;
}

struct FileAtLevel {
  std::unique_ptr<TFile> owned; // empty for level 0, which is the file we started from
  TFile* file = nullptr;        // nullptr if the chain of parents is shorter than the level
  std::string name;
};

// Follow the first parent file `level` times, starting from `file`
FileAtLevel openAtLevel(TFile* file, std::string const& filename, int level, ConfigParamRegistry& registry)
{
  FileAtLevel result{nullptr, file, filename};
  for (auto l = 0; l < level; ++l) {
    auto parentFiles = (TMap*)result.file->Get("parentFiles");
    if (!parentFiles || parentFiles->GetSize() == 0) {
      LOGP(info, "No parent file found for \"{}\"", result.name);
      return {nullptr, nullptr, result.name};
    }
    std::string parentFilename;
    for (auto* p : *parentFiles) {
      parentFilename = ((TPair*)p)->Value()->GetName();
      break;
    }
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
      pos = parentFilename.find(from);
      if (pos != std::string::npos) {
        parentFilename.replace(pos, from.length(), to);
      }
    }

    if (parentFilename.starts_with("alien://") && !gGrid) {
      TGrid::Connect("alien://");
    }

    std::unique_ptr<TFile> parentFile{TFile::Open(parentFilename.c_str())};
    if (parentFile.get() == nullptr) {
      LOGP(fatal, "Couldn't open derived file \"{}\"!", parentFilename);
    }
    result.file = parentFile.get();
    result.owned = std::move(parentFile);
    result.name = parentFilename;
  }
  return result;
}

// What was found at a given parent level
struct LevelMetadata {
  bool found = false; // the file at this level exists
  std::string source;
  std::vector<std::pair<std::string, std::string>> metadata;
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
        std::unique_ptr<TFile> currentFile{TFile::Open(filename.c_str())};
        if (currentFile.get() == nullptr) {
          LOGP(fatal, "Couldn't open file \"{}\"!", filename);
        }

        // Each level is read only once, even if several origins map to it
        std::map<int, LevelMetadata> levels;
        auto readLevel = [&](int level) -> LevelMetadata const& {
          if (auto it = levels.find(level); it != levels.end()) {
            return it->second;
          }
          LevelMetadata info;
          auto atLevel = openAtLevel(currentFile.get(), filename, level, registry);
          if (atLevel.file) {
            info.found = true;
            info.source = atLevel.name;
            info.metadata = readMetadata(atLevel.file);
            info.tables = getListOfTables(atLevel.file);
          }
          return levels.emplace(level, std::move(info)).first->second;
        };
        int parentAccessLevel = 0;
        if (registry.isSet("aod-parent-access-level")) {
          parentAccessLevel = registry.get<int>("aod-parent-access-level");
        }

        // The plain aod-metadata-* options, from the file itself or from its first parent
        auto const& top = readLevel(0);
        std::vector<ConfigParamSpec> results = metadataToSpecs(top.metadata);
        std::vector<std::string> plainKeys;
        if (!top.tables.empty()) {
          results.push_back(ConfigParamSpec{"aod-metadata-tables", VariantType::ArrayString, top.tables, {"Tables in first AOD"}});
        }
        if (!top.metadata.empty()) {
          // Found metadata already in the main file.
          results.push_back(ConfigParamSpec{"aod-metadata-source", VariantType::String, filename, {"File from which the metadata was extracted."}});
          plainKeys = metadataKeys(top.metadata);
        } else if (parentAccessLevel == 0) {
          LOGP(info, "No metadata found in file \"{}\" and parent level 0 prevents further lookup.", filename);
          results.push_back(ConfigParamSpec{"aod-metadata-disable", VariantType::String, "1", {"Metadata not found in AOD"}});
        } else {
          // Lets try in parent file.
          LOGP(info, "No metadata found in file \"{}\", checking in its parents.", filename);
          auto const& parent = readLevel(1);
          if (!parent.found) {
            LOGP(info, "No metadata found in file \"{}\"", filename);
          } else {
            results = metadataToSpecs(parent.metadata);
            if (!parent.metadata.empty()) {
              if (!parent.tables.empty()) {
                results.push_back(ConfigParamSpec{"aod-metadata-tables", VariantType::ArrayString, parent.tables, {"Tables in first AOD"}});
              }
              results.push_back(ConfigParamSpec{"aod-metadata-source", VariantType::String, filename, {"File from which the metadata was extracted."}});
              plainKeys = metadataKeys(parent.metadata);
            } else {
              LOGP(info, "No metadata found in file \"{}\" nor in its parent file \"{}\"", filename, parent.source);
            }
          }
          if (plainKeys.empty()) {
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
        std::vector<std::string> keys = plainKeys;
        for (auto i = 0U; i < levelMap.rows(); ++i) {
          auto level = levelMap.get(i, 0U);
          if (level > parentAccessLevel) {
            LOGP(fatal, "Origin {} is mapped to parent level {}, but aod-parent-access-level is {}", origins[i], level, parentAccessLevel);
          }
          for (auto const& [key, value] : readLevel(level).metadata) {
            if (std::ranges::find(keys, key) == keys.end()) {
              keys.push_back(key);
            }
          }
        }
        auto sourceColumn = keys.size();
        keys.emplace_back("source");
        auto disableColumn = keys.size();
        keys.emplace_back("disable");

        std::vector<std::string> values(origins.size() * keys.size());
        for (auto i = 0U; i < origins.size(); ++i) {
          auto const& info = readLevel(levelMap.get(i, 0U));
          auto* row = values.data() + i * keys.size();
          for (auto const& [key, value] : info.metadata) {
            row[std::ranges::find(keys, key) - keys.begin()] = value;
          }
          row[sourceColumn] = info.source;
          if (info.metadata.empty()) {
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
