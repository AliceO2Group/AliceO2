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
// O2 includes

#include "Framework/AnalysisTask.h"
#include "Framework/AnalysisDataModel.h"
#include "Framework/ASoAHelpers.h"
#include "AnalysisDataModel/TrackSelectionTables.h"

#include "filterTables.h"

#include "Framework/HistogramRegistry.h"

#include <iostream>
#include <cstdio>
#include <random>
#include <fmt/format.h>
#include <rapidjson/document.h>
#include <rapidjson/filereadstream.h>

// we need to add workflow options before including Framework/runDataProcessing
void customize(std::vector<o2::framework::ConfigParamSpec>& workflowOptions)
{
  // option allowing to set parameters
  std::vector<o2::framework::ConfigParamSpec> options{o2::framework::ConfigParamSpec{"train_config", o2::framework::VariantType::String, "full_config.json", {"Configuration of the filtering train"}}};
  std::unordered_map<std::string, std::unordered_map<std::string, float>> downscalings;
  o2::aod::FillFiltersMap(o2::aod::FiltersPack, downscalings);
  for (auto& table : downscalings) {
    for (auto& col : table.second) {
      options.emplace_back(o2::framework::ConfigParamSpec{table.first + "_" + col.first, o2::framework::VariantType::Float, 1.f, {std::string("Downscaling for the filter ") + col.first + " of the class " + table.first}});
    }
  }

  std::swap(workflowOptions, options);
}

#include "Framework/runDataProcessing.h"

using namespace o2;
using namespace o2::framework;
using namespace o2::framework::expressions;
using namespace rapidjson;

namespace
{
bool readJsonFile(std::string& config, Document& d)
{
  FILE* fp = fopen(config.data(), "rb");
  if (!fp) {
    LOG(ERROR) << "Missing configuration json file: " << config;
    return false;
  }

  char readBuffer[65536];
  FileReadStream is(fp, readBuffer, sizeof(readBuffer));

  d.ParseStream(is);
  fclose(fp);
  return true;
}

std::unordered_map<std::string, std::unordered_map<std::string, float>> mDownscaling;

} // namespace

struct centralEventFilterTask {

  HistogramRegistry scalers{"scalers", {}, OutputObjHandlingPolicy::AnalysisObject, true, true};

  void init(o2::framework::InitContext&)
  {
    LOG(INFO) << "Start init";
    int nCols{0};
    for (auto& table : mDownscaling) {
      nCols += table.second.size();
    }
    LOG(INFO) << "Middle init, total number of columns " << nCols;

    scalers.add("mScalers", "", HistType::kTH1F, {{nCols + 1, -0.5, 0.5 + nCols, ";;Number of events"}});
    scalers.add("mFiltered", "", HistType::kTH1F, {{nCols + 1, -0.5, 0.5 + nCols, ";;Number of filtered events"}});
    auto mScalers = scalers.get<TH1>(HIST("mScalers"));
    auto mFiltered = scalers.get<TH1>(HIST("mFiltered"));

    mScalers->GetXaxis()->SetBinLabel(1, "Total number of events");
    mFiltered->GetXaxis()->SetBinLabel(1, "Total number of events");
    int bin{2};
    for (auto& table : mDownscaling) {
      for (auto& column : table.second) {
        mScalers->GetXaxis()->SetBinLabel(bin, column.first.data());
        mFiltered->GetXaxis()->SetBinLabel(bin++, column.first.data());
      }
    }
  }

  void run(ProcessingContext& pc)
  {

    auto mScalers = scalers.get<TH1>(HIST("mScalers"));
    auto mFiltered = scalers.get<TH1>(HIST("mFiltered"));
    int64_t nEvents{-1};
    for (auto& tableName : mDownscaling) {
      auto tableConsumer = pc.inputs().get<TableConsumer>(tableName.first);

      auto tablePtr{tableConsumer->asArrowTable()};
      int64_t nRows{tablePtr->num_rows()};
      nEvents = nEvents < 0 ? nRows : nEvents;
      if (nEvents != nRows) {
        LOG(FATAL) << "Inconsistent number of rows across trigger tables.";
      }

      auto schema{tablePtr->schema()};
      for (auto& colName : tableName.second) {
        int bin{mScalers->GetXaxis()->FindBin(colName.first.data())};
        double binCenter{mScalers->GetXaxis()->GetBinCenter(bin)};
        auto column{tablePtr->GetColumnByName(colName.first)};
        double downscaling{colName.second};
        if (column) {
          for (int64_t iC{0}; iC < column->num_chunks(); ++iC) {
            auto chunk{column->chunk(iC)};
            auto boolArray = std::static_pointer_cast<arrow::BooleanArray>(chunk);
            for (int64_t iS{0}; iS < chunk->length(); ++iS) {
              if (boolArray->Value(iS)) {
                mScalers->Fill(binCenter);
                if (mUniformGenerator(mGeneratorEngine) < downscaling) {
                  mFiltered->Fill(binCenter);
                }
              }
            }
          }
        }
      }
    }
    mScalers->SetBinContent(1, mScalers->GetBinContent(1) + nEvents);
    mFiltered->SetBinContent(1, mFiltered->GetBinContent(1) + nEvents);
  }

  std::mt19937_64 mGeneratorEngine;
  std::uniform_real_distribution<double> mUniformGenerator = std::uniform_real_distribution<double>(0., 1.);
};

WorkflowSpec defineDataProcessing(ConfigContext const& cfg)
{
  std::vector<InputSpec> inputs;

  auto config = cfg.options().get<std::string>("train_config");
  Document d;
  std::unordered_map<std::string, std::unordered_map<std::string, float>> downscalings;
  o2::aod::FillFiltersMap(o2::aod::FiltersPack, downscalings);

  std::array<bool, o2::aod::NumberOfFilters> enabledFilters = {false};
  if (readJsonFile(config, d)) {
    for (auto& workflow : d["workflows"].GetArray()) {
      for (uint32_t iFilter{0}; iFilter < o2::aod::NumberOfFilters; ++iFilter) {
        if (std::string_view(workflow["workflow_name"].GetString()) == std::string_view(o2::aod::FilteringTaskNames[iFilter])) {
          inputs.emplace_back(std::string(o2::aod::AvailableFilters[iFilter]), "AOD", o2::aod::FilterDescriptions[iFilter], 0, Lifetime::Timeframe);
          enabledFilters[iFilter] = true;
          break;
        }
      }
    }
  }

  for (uint32_t iFilter{0}; iFilter < o2::aod::NumberOfFilters; ++iFilter) {
    if (!enabledFilters[iFilter]) {
      LOG(INFO) << std::string_view(o2::aod::AvailableFilters[iFilter]) << " not present in the configuration, removing it.";
      downscalings.erase(std::string(o2::aod::AvailableFilters[iFilter]));
    } else {
      std::string tableName{o2::aod::AvailableFilters[iFilter]};
      for (auto& cols : downscalings[tableName]) {
        std::string optName{tableName + "_" + cols.first};
        auto downscaling = cfg.options().get<float>(optName.data());
        cols.second = downscaling;
      }
    }
  }

  DataProcessorSpec spec{adaptAnalysisTask<centralEventFilterTask>(cfg)};
  for (auto& input : inputs) {
    spec.inputs.emplace_back(input);
  }
  mDownscaling.swap(downscalings);

  return WorkflowSpec{spec};
}
