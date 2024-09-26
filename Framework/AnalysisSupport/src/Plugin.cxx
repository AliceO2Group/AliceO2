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
#include <TFile.h>
#include <TMap.h>
#include <TGrid.h>
#include <TObjString.h>
#include <TString.h>
#include <fmt/format.h>

O2_DECLARE_DYNAMIC_LOG(analysis_support);

struct ROOTFileReader : o2::framework::AlgorithmPlugin {
  o2::framework::AlgorithmSpec create() override
  {
    return o2::framework::readers::AODJAlienReaderHelpers::rootFileReaderCallback();
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
        } },
      .kind = ServiceKind::Serial};
  }
};

std::vector<std::string> getListOfTables(TFile* f)
{
  std::vector<std::string> r;
  TList* keyList = f->GetListOfKeys();

  for (auto key : *keyList) {
    if (!std::string_view(key->GetName()).starts_with("DF_")) {
      continue;
    }
    auto* d = (TDirectory*)f->Get(key->GetName());
    TList* branchList = d->GetListOfKeys();
    for (auto b : *branchList) {
      r.emplace_back(b->GetName());
    }
    break;
  }
  return r;
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
        std::vector<ConfigParamSpec> results;
        TFile* currentFile = nullptr;
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
        if (filename.rfind("alien://", 0) == 0) {
          TGrid::Connect("alien://");
        }
        LOGP(info, "Loading metadata from file {} in PID {}", filename, getpid());
        currentFile = TFile::Open(filename.c_str());
        if (!currentFile) {
          LOGP(fatal, "Couldn't open file \"{}\"!", filename);
        }

        // Get the metadata, if any
        auto m = (TMap*)currentFile->Get("metaData");
        if (!m) {
          LOGP(info, "No metadata found in file \"{}\"", filename);
          results.push_back(ConfigParamSpec{"aod-metadata-disable", VariantType::String, "1", {"Metadata not found in AOD"}});
          return results;
        }
        auto it = m->MakeIterator();

        // Serialise metadata into a ; separated string with : separating key and value
        bool first = true;
        while (auto obj = it->Next()) {
          if (first) {
            LOGP(info, "Metadata for file \"{}\":", filename);
            first = false;
          }
          auto objString = (TObjString*)m->GetValue(obj);
          LOGP(info, "- {}: {}", obj->GetName(), objString->String().Data());
          std::string key = "aod-metadata-" + std::string(obj->GetName());
          char const* value = strdup(objString->String());
          results.push_back(ConfigParamSpec{key, VariantType::String, value, {"Metadata in AOD"}});
        }

        auto tables = getListOfTables(currentFile);
        if (tables.empty() == false) {
          results.push_back(ConfigParamSpec{"aod-metadata-tables", VariantType::ArrayString, tables, {"Tables in first AOD"}});
        }
        return results;
      }};
  }
};

DEFINE_DPL_PLUGINS_BEGIN
DEFINE_DPL_PLUGIN_INSTANCE(ROOTFileReader, CustomAlgorithm);
DEFINE_DPL_PLUGIN_INSTANCE(RunSummary, CustomService);
DEFINE_DPL_PLUGIN_INSTANCE(DiscoverMetadataInAOD, ConfigDiscovery);
DEFINE_DPL_PLUGINS_END
