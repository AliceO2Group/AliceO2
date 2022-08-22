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

/// \file  emc-channel-data-producer.cxx
/// \brief This task generates an emcal event with a number of cells. It is designed to produce data for the testing of the bad channel and time calibration of the emcal
///
/// \author  Joshua Koenig <joshua.konig@cern.ch>

#include "DataFormatsEMCAL/Cell.h"
#include "DataFormatsEMCAL/TriggerRecord.h"
#include <Framework/DataProcessorSpec.h>

#include "Framework/Task.h"
#include "Framework/WorkflowSpec.h"

#include "Framework/DataRefUtils.h"
#include "Framework/ControlService.h"
#include "Algorithm/RangeTokenizer.h"
#include <random>

#include "TFile.h"
#include "TH2.h"
#include "TH1.h"

#include <unistd.h>

using namespace o2::framework;

DataProcessorSpec generateData(const std::string nameRootFile, const std::string nameInputHist, const bool isTimeCalib, const int nCellsPerEvent);

// we need to add workflow options before including Framework/runDataProcessing
void customize(std::vector<ConfigParamSpec>& workflowOptions)
{
  std::vector<ConfigParamSpec> options{
    {"inputRootFile", VariantType::String, "", {"input root file from which data is taken, if empty, random data will be produced"}},
    {"nameInputHist", VariantType::String, "", {"name of the 2d histogram inside the root file used for the data generation"}},
    {"isInputTimeCalib", VariantType::Bool, false, {"input is produced for time clibration or bad channel calibration. Information is needed if inputRootFiel is specified as it has ifferent content for bad channel calib and time calib"}},
    {"nCellsPerEvent", VariantType::Int, 5, {"number of cells per emcal triggered event"}}};

  std::swap(workflowOptions, options);
}

#include "Framework/runDataProcessing.h"

WorkflowSpec defineDataProcessing(ConfigContext const& config)
{

  const std::string nameRootFile = config.options().get<std::string>("inputRootFile");
  const std::string nameInputHist = config.options().get<std::string>("nameInputHist");
  const bool isTimeCalib = config.options().get<bool>("isInputTimeCalib");
  const int nCellsPerEvent = config.options().get<int>("nCellsPerEvent");

  WorkflowSpec workflow;
  workflow.emplace_back(generateData(nameRootFile, nameInputHist, isTimeCalib, nCellsPerEvent));

  return workflow;
}

DataProcessorSpec generateData(const std::string nameRootFile, const std::string nameInputHist, const bool isTimeCalib, const int nCellsPerEvent)
{
  std::vector<OutputSpec> outputSpecs;
  outputSpecs.emplace_back(o2::header::gDataOriginEMC, "CELLS", 0, o2::framework::Lifetime::Timeframe);
  outputSpecs.emplace_back(o2::header::gDataOriginEMC, "CELLSTRGR", 0, o2::framework::Lifetime::Timeframe);

  // initialize random number generators to generate cell time, cell energy and cellID
  std::default_random_engine generator;
  std::uniform_real_distribution<> disCellID(0, 17663);
  std::exponential_distribution<float> disEnergy(3.5);
  std::normal_distribution<float> disTime(20, 10);

  // load 2d root file in case the paths are specified
  TH2F* h2d = nullptr;
  if (nameRootFile.find(".root") != std::string::npos) {
    TFile* f = TFile::Open(nameRootFile.c_str());
    if (!f) {
      LOG(error) << "root file does not exist " << nameRootFile;
    }
    h2d = (TH2F*)f->Get(nameInputHist.c_str());
    h2d->SetDirectory(nullptr);
    if (!h2d) {
      LOG(error) << "histogram does not exist " << nameInputHist;
    }
  }

  return DataProcessorSpec{
    "emcal-cell-data-producer",
    Inputs{},
    outputSpecs,
    AlgorithmSpec{
      [=](ProcessingContext& ctx) mutable {
        o2::pmr::vector<o2::emcal::Cell> CellOutput;
        for (int i = 0; i < nCellsPerEvent; ++i) {
          double cellID = 0;
          double cellE = 0;
          double cellTime = 0;
          if (h2d) {
            // case for time calibration
            if (isTimeCalib) {
              h2d->GetRandom2(cellTime, cellID);
              cellE = disEnergy(generator);
            } else {
              h2d->GetRandom2(cellE, cellID);
              cellTime = disTime(generator);
            }
          } else {
            cellID = disCellID(generator);
            cellE = disEnergy(generator);
            cellTime = disTime(generator);
          }
          // for now only consider low gain cells. Maybe implement high gain cells
          CellOutput.emplace_back(static_cast<int>(cellID), cellE, cellTime, o2::emcal::ChannelType_t::LOW_GAIN);
        }
        LOG(debug) << "sending " << CellOutput.size() << "cells";
        o2::pmr::vector<o2::emcal::TriggerRecord> TriggerOutput;
        TriggerOutput.emplace_back(0, 0, 0, CellOutput.size());

        ctx.outputs().adoptContainer(Output{o2::header::gDataOriginEMC, "CELLS", 0, Lifetime::Timeframe}, std::move(CellOutput));
        ctx.outputs().adoptContainer(Output{o2::header::gDataOriginEMC, "CELLSTRGR", 0, Lifetime::Timeframe}, std::move(TriggerOutput));
      }}};
}
