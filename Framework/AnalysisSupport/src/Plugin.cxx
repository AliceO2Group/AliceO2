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

std::vector<std::string> getListOfTables(std::unique_ptr<TFile>& f)
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

auto readMetadata(std::unique_ptr<TFile>& currentFile) -> std::vector<ConfigParamSpec>
{
  // Get the metadata, if any
  auto m = (TMap*)currentFile->Get("metaData");
  if (!m) {
    return {};
  }
  std::vector<ConfigParamSpec> results;
  auto it = m->MakeIterator();

  // Serialise metadata into a ; separated string with : separating key and value
  bool first = true;
  while (auto obj = it->Next()) {
    if (first) {
      LOGP(info, "Metadata for file \"{}\":", currentFile->GetName());
      first = false;
    }
    auto objString = (TObjString*)m->GetValue(obj);
    LOGP(info, "- {}: {}", obj->GetName(), objString->String().Data());
    std::string key = "aod-metadata-" + std::string(obj->GetName());
    char const* value = strdup(objString->String());
    results.push_back(ConfigParamSpec{key, VariantType::String, value, {"Metadata in AOD"}});
  }

  return results;
}

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
        std::vector<ConfigParamSpec> results = readMetadata(currentFile);
        const bool metaDataEmpty = results.empty();
        auto tables = getListOfTables(currentFile);
        if (tables.empty() == false) {
          results.push_back(ConfigParamSpec{"aod-metadata-tables", VariantType::ArrayString, tables, {"Tables in first AOD"}});
        }

        // Found metadata already in the main file.
        if (!metaDataEmpty) {
          results.push_back(ConfigParamSpec{"aod-metadata-source", VariantType::String, filename, {"File from which the metadata was extracted."}});
          return results;
        }

        if (!registry.isSet("aod-parent-access-level") || registry.get<int>("aod-parent-access-level") == 0) {
          LOGP(info, "No metadata found in file \"{}\" and parent level 0 prevents further lookup.", filename);
          results.push_back(ConfigParamSpec{"aod-metadata-disable", VariantType::String, "1", {"Metadata not found in AOD"}});
          return results;
        }

        // Lets try in parent file.
        auto parentFiles = (TMap*)currentFile->Get("parentFiles");
        if (!parentFiles) {
          LOGP(info, "No metadata found in file \"{}\"", filename);
          results.push_back(ConfigParamSpec{"aod-metadata-disable", VariantType::String, "1", {"Metadata not found in AOD"}});
          return results;
        }
        LOGP(info, "No metadata found in file \"{}\", checking in its parents.", filename);
        for (auto* p : *parentFiles) {
          std::string parentFilename = ((TPair*)p)->Value()->GetName();
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
          results = readMetadata(parentFile);
          // Found metadata already in the main file.
          if (!results.empty()) {
            auto tables = getListOfTables(parentFile);
            if (tables.empty() == false) {
              results.push_back(ConfigParamSpec{"aod-metadata-tables", VariantType::ArrayString, tables, {"Tables in first AOD"}});
            }
            results.push_back(ConfigParamSpec{"aod-metadata-source", VariantType::String, filename, {"File from which the metadata was extracted."}});
            return results;
          }
          LOGP(info, "No metadata found in file \"{}\" nor in its parent file \"{}\"", filename, parentFilename);
          break;
        }
        results.push_back(ConfigParamSpec{"aod-metadata-disable", VariantType::String, "1", {"Metadata not found in AOD"}});
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
