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

/// @file   centralEventFilterProcessor.cxx

#include "centralEventFilterProcessor.h"

#include "Framework/ControlService.h"
#include "Framework/ConfigParamRegistry.h"
#include <Framework/Logger.h>
#include "Framework/TableConsumer.h"

#include <TFile.h>
#include <TH1D.h>

#include <iostream>
#include <cstdio>
#include <fmt/format.h>
#include <rapidjson/document.h>
#include <rapidjson/filereadstream.h>

using namespace o2::framework;
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
} // namespace

namespace o2::aod::filtering
{

void CentralEventFilterProcessor::init(framework::InitContext& ic)
{
  // JSON example
  // {
  //   "subwagon_name" : "CentralEventFilterProcessor",
  //   "configuration" : {
  //     "NucleiFilters" : {
  //       "H2" : 0.1,
  //       "H3" : 0.3,
  //       "HE3" : 1.,
  //       "HE4" : 1.
  //     }
  //   }
  // }
  LOG(INFO) << "Start init";
  Document d;
  int nCols{0};
  if (readJsonFile(mConfigFile, d)) {
    for (auto& workflow : d["workflows"].GetArray()) {
      if (std::string_view(workflow["subwagon_name"].GetString()) == "CentralEventFilterProcessor") {
        auto& config = workflow["configuration"];
        for (auto& filter : AvailableFilters) {
          auto& filterConfig = config[filter];
          if (filterConfig.IsObject()) {
            std::unordered_map<std::string, float> tableMap;
            for (auto& node : filterConfig.GetObject()) {
              tableMap[node.name.GetString()] = node.value.GetDouble();
              nCols++;
            }
            LOG(INFO) << "Enabling downscaling map for filter: " << filter;
            mDownscaling[filter] = tableMap;
          }
        }
        break;
      }
    }
  }
  LOG(INFO) << "Middle init" << std::endl;
  mScalers = new TH1D("mScalers", ";;Number of events", nCols + 1, -0.5, 0.5 + nCols);
  mScalers->GetXaxis()->SetBinLabel(1, "Total number of events");

  mFiltered = new TH1D("mFiltered", ";;Number of filtered events", nCols + 1, -0.5, 0.5 + nCols);
  mFiltered->GetXaxis()->SetBinLabel(1, "Total number of events");

  int bin{2};
  for (auto& table : mDownscaling) {
    for (auto& column : table.second) {
      mScalers->GetXaxis()->SetBinLabel(bin, column.first.data());
      mFiltered->GetXaxis()->SetBinLabel(bin++, column.first.data());
    }
  }

  TFile test("test.root", "recreate");
  mScalers->Clone()->Write();
  test.Close();
}

void CentralEventFilterProcessor::run(ProcessingContext& pc)
{
  int64_t nEvents{-1};
  for (auto& tableName : mDownscaling) {
    auto tableConsumer = pc.inputs().get<TableConsumer>(tableName.first);

    auto tablePtr{tableConsumer->asArrowTable()};
    int64_t nRows{tablePtr->num_rows()};
    nEvents = nEvents < 0 ? nRows : nEvents;
    if (nEvents != nRows) {
      LOG(FATAL) << "Inconsistent number of rows across trigger tables, fatal" << std::endl; ///TODO: move it to real fatal
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

void CentralEventFilterProcessor::endOfStream(EndOfStreamContext& ec)
{
  TFile output("trigger.root", "recreate");
  mScalers->Write("Scalers");
  mFiltered->Write("FilteredScalers");
  if (mScalers->GetBinContent(1) > 1.e-24) {
    mScalers->Scale(1. / mScalers->GetBinContent(1));
  }
  if (mFiltered->GetBinContent(1) > 1.e-24) {
    mFiltered->Scale(1. / mFiltered->GetBinContent(1));
  }
  mScalers->Write("Fractions");
  mFiltered->Write("FractionsDownscaled");
  output.Close();
}

DataProcessorSpec getCentralEventFilterProcessorSpec(std::string& config)
{

  std::vector<InputSpec> inputs;
  Document d;

  if (readJsonFile(config, d)) {
    for (auto& workflow : d["workflows"].GetArray()) {
      for (unsigned int iFilter{0}; iFilter < AvailableFilters.size(); ++iFilter) {
        if (std::string_view(workflow["subwagon_name"].GetString()) == std::string_view(AvailableFilters[iFilter])) {
          inputs.emplace_back(std::string(AvailableFilters[iFilter]), "AOD", FilterDescriptions[iFilter], 0, Lifetime::Timeframe);
          LOG(INFO) << "Adding input " << std::string_view(AvailableFilters[iFilter]) << std::endl;
          break;
        }
      }
    }
  }

  std::vector<OutputSpec> outputs;
  outputs.emplace_back("AOD", "Decision", 0, Lifetime::Timeframe);
  outputs.emplace_back("TFN", "TFNumber", 0, Lifetime::Timeframe);

  return DataProcessorSpec{
    "o2-central-event-filter-processor",
    inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<CentralEventFilterProcessor>(config)},
    Options{
      {"filtering-config", VariantType::String, "", {"Path to the filtering json config file"}}}};
}

} // namespace o2::aod::filtering
