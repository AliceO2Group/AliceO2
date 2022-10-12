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

#include <algorithm>
#include <filesystem>
#include <gsl/span>
#include "DataFormatsEMCAL/Cell.h"
#include "DataFormatsEMCAL/TriggerRecord.h"
#include "EMCALCalib/BadChannelMap.h"
#include "EMCALCalib/TimeCalibrationParams.h"
#include "EMCALCalib/GainCalibrationFactors.h"
#include "EMCALWorkflow/CellRecalibratorSpec.h"
#include "Framework/CCDBParamSpec.h"
#include <TFile.h>

using namespace o2::emcal;

CellRecalibratorSpec::CellRecalibratorSpec(uint32_t outputspec, bool badChannelCalib, bool timeCalib, bool gainCalib, std::shared_ptr<o2::emcal::CalibLoader> calibHandler) : mOutputSubspec(outputspec), mCalibrationHandler(calibHandler)
{
  setRunBadChannelCalibration(badChannelCalib);
  setRunTimeCalibration(timeCalib);
  setRunGainCalibration(gainCalib);
}

void CellRecalibratorSpec::init(framework::InitContext& ctx)
{
  // Display calibrations which are enabled:
  LOG(info) << "Bad channel calibration: " << (isRunBadChannlCalibration() ? "enabled" : "disabled");
  LOG(info) << "Time calibration:        " << (isRunTimeCalibration() ? "enabled" : "disabled");
  LOG(info) << "Gain calibration:        " << (isRunGainCalibration() ? "enabled" : "disabled");
}

void CellRecalibratorSpec::run(framework::ProcessingContext& ctx)
{
  auto inputcells = ctx.inputs().get<gsl::span<o2::emcal::Cell>>("cells");
  auto intputtriggers = ctx.inputs().get<gsl::span<o2::emcal::TriggerRecord>>("triggerrecords");
  LOG(info) << "Received " << inputcells.size() << " cells from " << intputtriggers.size() << " triggers";

  mCalibrationHandler->checkUpdates(ctx);
  updateCalibObjects();

  std::vector<o2::emcal::Cell> outputcells;
  std::vector<o2::emcal::TriggerRecord> outputtriggers;

  uint32_t currentfirst = outputcells.size();
  for (const auto& trg : intputtriggers) {
    if (!trg.getNumberOfObjects()) {
      outputtriggers.emplace_back(trg.getBCData(), currentfirst, trg.getNumberOfObjects()).setTriggerBits(trg.getTriggerBits());
      continue;
    }
    auto calibratedCells = mCellRecalibrator.getCalibratedCells(gsl::span<const o2::emcal::Cell>(inputcells.data() + trg.getFirstEntry(), trg.getNumberOfObjects()));
    if (calibratedCells.size()) {
      std::copy(calibratedCells.begin(), calibratedCells.end(), std::back_insert_iterator(outputcells));
    }
    outputtriggers.emplace_back(trg.getBCData(), currentfirst, calibratedCells.size()).setTriggerBits(trg.getTriggerBits());
    currentfirst = outputcells.size();
  }

  LOG(info) << "Timeframe: " << inputcells.size() << " cells read, " << outputcells.size() << " cells kept";

  // send recalibrated objects
  ctx.outputs().snapshot(o2::framework::Output{o2::header::gDataOriginEMC, "CELLS", mOutputSubspec, o2::framework::Lifetime::Timeframe}, outputcells);
  ctx.outputs().snapshot(o2::framework::Output{o2::header::gDataOriginEMC, "CELLSTRGR", mOutputSubspec, o2::framework::Lifetime::Timeframe}, outputtriggers);
}

void CellRecalibratorSpec::finaliseCCDB(o2::framework::ConcreteDataMatcher& matcher, void* obj)
{
  LOG(info) << "Handling new Calibration objects";
  if (mCalibrationHandler->finalizeCCDB(matcher, obj)) {
    return;
  }
}

void CellRecalibratorSpec::updateCalibObjects()
{
  if (isRunBadChannlCalibration()) {
    if (mCalibrationHandler->hasUpdateBadChannelMap()) {
      LOG(info) << "updateCalibObjects: Bad channel map changed";
      mCellRecalibrator.setBadChannelMap(mCalibrationHandler->getBadChannelMap());
    }
  }
  if (isRunTimeCalibration()) {
    if (mCalibrationHandler->hasUpdateTimeCalib()) {
      LOG(info) << "updateCalibObjects: Time calib params changed";
      mCellRecalibrator.setTimeCalibration(mCalibrationHandler->getTimeCalibration());
    }
  }
  if (isRunGainCalibration()) {
    if (mCalibrationHandler->hasUpdateGainCalib()) {
      LOG(info) << "updateCalibObjects: Time calib params changed";
      mCellRecalibrator.setGainCalibration(mCalibrationHandler->getGainCalibration());
    }
  }
}

o2::framework::DataProcessorSpec o2::emcal::getCellRecalibratorSpec(uint32_t inputSubspec, uint32_t outputSubspec, bool badChannelCalib, bool timeCalib, bool gainCalib)
{
  auto calibhandler = std::make_shared<o2::emcal::CalibLoader>();
  calibhandler->enableBadChannelMap(badChannelCalib);
  calibhandler->enableTimeCalib(timeCalib);
  calibhandler->enableGainCalib(gainCalib);
  std::vector<o2::framework::InputSpec>
    inputs = {{"cells", o2::header::gDataOriginEMC, "CELLS", inputSubspec, o2::framework::Lifetime::Timeframe},
              {"triggerrecords", o2::header::gDataOriginEMC, "CELLSTRGR", inputSubspec, o2::framework::Lifetime::Timeframe}};
  std::vector<o2::framework::OutputSpec> outputs = {{o2::header::gDataOriginEMC, "CELLS", outputSubspec, o2::framework::Lifetime::Timeframe},
                                                    {o2::header::gDataOriginEMC, "CELLSTRGR", outputSubspec, o2::framework::Lifetime::Timeframe}};
  calibhandler->defineInputSpecs(inputs);

  return o2::framework::DataProcessorSpec{"EMCALCellRecalibrator",
                                          inputs,
                                          outputs,
                                          o2::framework::adaptFromTask<o2::emcal::CellRecalibratorSpec>(outputSubspec, badChannelCalib, timeCalib, gainCalib, calibhandler)};
}
