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

  mCalibrationHandler->static_load();
}

void CellRecalibratorSpec::run(framework::ProcessingContext& ctx)
{
  auto inputcells = ctx.inputs().get<gsl::span<o2::emcal::Cell>>("cells");
  auto intputtriggers = ctx.inputs().get<gsl::span<o2::emcal::TriggerRecord>>("triggerrecords");

  updateCalibObjects();

  std::vector<o2::emcal::Cell> outputcells;
  std::vector<o2::emcal::TriggerRecord> outputtriggers;

  uint32_t currentfirst = outputcells.size();
  for (const auto& trg : intputtriggers) {
    if (!trg.getNumberOfObjects()) {
      o2::emcal::TriggerRecord nexttrigger(trg.getBCData(), currentfirst, trg.getNumberOfObjects());
      nexttrigger.setTriggerBits(trg.getTriggerBits());
      outputtriggers.push_back(nexttrigger);
      continue;
    }
    uint32_t cellsEvent = 0;
    for (const auto& cell : gsl::span<const o2::emcal::Cell>(inputcells.data() + trg.getFirstEntry(), trg.getNumberOfObjects())) {
      auto calibrated = getCalibratedCell(cell);
      if (calibrated) {
        outputcells.push_back(calibrated.value());
        cellsEvent++;
      }
    }
    o2::emcal::TriggerRecord nexttrigger(trg.getBCData(), currentfirst, cellsEvent);
    nexttrigger.setTriggerBits(trg.getTriggerBits());
    outputtriggers.push_back(nexttrigger);
    currentfirst = outputcells.size();
  }

  // send recalibrated objects
  ctx.outputs().snapshot(o2::framework::Output{o2::header::gDataOriginEMC, "CELLS", mOutputSubspec, o2::framework::Lifetime::Timeframe}, outputcells);
  ctx.outputs().snapshot(o2::framework::Output{o2::header::gDataOriginEMC, "CELLSTRGR", mOutputSubspec, o2::framework::Lifetime::Timeframe}, outputtriggers);
}

void CellRecalibratorSpec::finaliseCCDB(o2::framework::ConcreteDataMatcher& matcher, void* obj)
{
  if (mCalibrationHandler->finalizeCCDB(matcher, obj)) {
    return;
  }
}

std::optional<o2::emcal::Cell> CellRecalibratorSpec::getCalibratedCell(const o2::emcal::Cell& input) const
{
  if (isRunBadChannlCalibration() && mBadChannelMap) {
    auto cellstatus = mBadChannelMap->getChannelStatus(input.getTower());
    // reject bad and dead cells
    if (cellstatus == BadChannelMap::MaskType_t::BAD_CELL || cellstatus == BadChannelMap::MaskType_t::DEAD_CELL) {
      return std::optional<o2::emcal::Cell>();
    }
  }
  auto celltime = input.getTimeStamp();
  auto cellenergy = input.getEnergy();
  if (isRunTimeCalibration() && mTimeCalibration) {
    celltime -= mTimeCalibration->getTimeCalibParam(input.getTower(), input.getLowGain());
  }
  if (isRunGainCalibration() && mGainCalibration) {
    cellenergy *= mGainCalibration->getGainCalibFactors(input.getTower());
  }
  o2::emcal::Cell outputcell(input.getTower(), cellenergy, celltime, input.getType());
  return std::optional<o2::emcal::Cell>(outputcell);
}

void CellRecalibratorSpec::updateCalibObjects()
{
  if (isRunBadChannlCalibration()) {
    mBadChannelMap = mCalibrationHandler->getBadChannelMap();
  }
  if (isRunTimeCalibration()) {
    mTimeCalibration = mCalibrationHandler->getTimeCalibration();
  }
  if (isRunGainCalibration()) {
    mGainCalibration = mCalibrationHandler->getGainCalibration();
  }
}

o2::framework::DataProcessorSpec o2::emcal::getCellRecalibratorSpec(uint32_t inputSubspec, uint32_t outputSubspec, bool badChannelCalib, bool timeCalib, bool gainCalib, const std::string_view pathBadChannelMap, const std::string_view pathTimeCalib, const std::string_view pathGainCalib)
{
  auto calibhandler = std::make_shared<o2::emcal::CalibLoader>();
  calibhandler->enableBadChannelMap(badChannelCalib);
  calibhandler->enableTimeCalib(timeCalib);
  calibhandler->enableGainCalib(gainCalib);
  if (pathBadChannelMap.length()) {
    calibhandler->setLoadBadChannelMapFromFile(pathBadChannelMap);
  }
  if (pathTimeCalib.length()) {
    calibhandler->setLoadTimeCalibFromFile(pathTimeCalib);
  }
  if (pathGainCalib.length()) {
    calibhandler->setLoadGainCalibFromFile(pathGainCalib);
  }
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
