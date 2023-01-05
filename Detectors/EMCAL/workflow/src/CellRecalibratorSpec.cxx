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
#include "CommonConstants/Triggers.h"
#include "DataFormatsEMCAL/Cell.h"
#include "DataFormatsEMCAL/TriggerRecord.h"
#include "EMCALCalib/BadChannelMap.h"
#include "EMCALCalib/TimeCalibrationParams.h"
#include "EMCALCalib/GainCalibrationFactors.h"
#include "EMCALWorkflow/CellRecalibratorSpec.h"
#include "Framework/CCDBParamSpec.h"

using namespace o2::emcal;

CellRecalibratorSpec::CellRecalibratorSpec(uint32_t outputspec, LEDEventSettings ledsettings, bool badChannelCalib, bool timeCalib, bool gainCalib, std::shared_ptr<o2::emcal::CalibLoader> calibHandler) : mOutputSubspec(outputspec), mLEDsettings(ledsettings), mCalibrationHandler(calibHandler)
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
  std::string ledsettingstitle;
  switch (mLEDsettings) {
    case LEDEventSettings::KEEP:
      ledsettingstitle = "keep";
      break;
    case LEDEventSettings::DROP:
      ledsettingstitle = "drop";
      break;
    case LEDEventSettings::REDIRECT:
      ledsettingstitle = "redirect to EMC/CELLS/10";
      break;
  };
  LOG(info) << "Handling of LED events: " << ledsettingstitle;
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

  // Prepare containers for LED triggers (in case they should be redirected to a different output)
  std::vector<o2::emcal::Cell> ledcells;
  std::vector<o2::emcal::TriggerRecord> ledtriggers;

  uint32_t currentfirst = outputcells.size(),
           currentledfirst = ledcells.size();
  for (const auto& trg : intputtriggers) {
    if (trg.getTriggerBits() & o2::trigger::Cal) {
      switch (mLEDsettings) {
        case LEDEventSettings::KEEP:
          // LED events stay always uncalibrated
          writeTrigger(inputcells.subspan(trg.getFirstEntry(), trg.getNumberOfObjects()), trg, outputcells, outputtriggers);
          continue;
        case LEDEventSettings::DROP:
          // LED events will be dropped, simply discard them
          continue;
        case LEDEventSettings::REDIRECT: {
          // LED events stay always uncalibrated
          writeTrigger(inputcells.subspan(trg.getFirstEntry(), trg.getNumberOfObjects()), trg, ledcells, ledtriggers);
          continue;
        }
      };
    }
    if (!trg.getNumberOfObjects()) {
      outputtriggers.emplace_back(trg.getBCData(), outputcells.size(), trg.getNumberOfObjects()).setTriggerBits(trg.getTriggerBits());
      continue;
    }
    auto calibratedCells = mCellRecalibrator.getCalibratedCells(gsl::span<const o2::emcal::Cell>(inputcells.data() + trg.getFirstEntry(), trg.getNumberOfObjects()));
    writeTrigger(calibratedCells, trg, outputcells, outputtriggers);
  }

  LOG(info) << "Timeframe: " << inputcells.size() << " cells read, " << outputcells.size() << " cells kept";
  if (mLEDsettings == LEDEventSettings::REDIRECT) {
    LOG(info) << "Redirecting " << ledcells.size() << " LED cells from " << ledtriggers.size() << " LED triggers";
  }

  // send recalibrated objects
  ctx.outputs().snapshot(o2::framework::Output{o2::header::gDataOriginEMC, "CELLS", mOutputSubspec, o2::framework::Lifetime::Timeframe}, outputcells);
  ctx.outputs().snapshot(o2::framework::Output{o2::header::gDataOriginEMC, "CELLSTRGR", mOutputSubspec, o2::framework::Lifetime::Timeframe}, outputtriggers);
  if (mLEDsettings == LEDEventSettings::REDIRECT) {
    ctx.outputs().snapshot(o2::framework::Output{o2::header::gDataOriginEMC, "CELLS", 10, o2::framework::Lifetime::Timeframe}, ledcells);
    ctx.outputs().snapshot(o2::framework::Output{o2::header::gDataOriginEMC, "CELLSTRGR", 10, o2::framework::Lifetime::Timeframe}, ledtriggers);
  }
}

void CellRecalibratorSpec::writeTrigger(const gsl::span<const o2::emcal::Cell> selectedCells, const o2::emcal::TriggerRecord& currenttrigger, std::vector<o2::emcal::Cell>& outputcontainer, std::vector<o2::emcal::TriggerRecord>& outputtriggers)
{
  std::size_t currentfirst = outputcontainer.size();
  if (selectedCells.size()) {
    std::copy(selectedCells.begin(), selectedCells.end(), std::back_inserter(outputcontainer));
  }
  outputtriggers.emplace_back(currenttrigger.getBCData(), currentfirst, selectedCells.size()).setTriggerBits(currenttrigger.getTriggerBits());
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

o2::framework::DataProcessorSpec o2::emcal::getCellRecalibratorSpec(uint32_t inputSubspec, uint32_t outputSubspec, uint32_t ledsettings, bool badChannelCalib, bool timeCalib, bool gainCalib)
{
  auto calibhandler = std::make_shared<o2::emcal::CalibLoader>();
  calibhandler->enableBadChannelMap(badChannelCalib);
  calibhandler->enableTimeCalib(timeCalib);
  calibhandler->enableGainCalib(gainCalib);
  std::vector<o2::framework::InputSpec>
    inputs = {{"cells", o2::header::gDataOriginEMC, "CELLS", inputSubspec, o2::framework::Lifetime::Timeframe},
              {"triggerrecords", o2::header::gDataOriginEMC, "CELLSTRGR", inputSubspec, o2::framework::Lifetime::Timeframe}};
  CellRecalibratorSpec::LEDEventSettings taskledsettings = CellRecalibratorSpec::LEDEventSettings::KEEP;
  switch (ledsettings) {
    case 0:
      taskledsettings = CellRecalibratorSpec::LEDEventSettings::KEEP;
      break;
    case 1:
      taskledsettings = CellRecalibratorSpec::LEDEventSettings::DROP;
      break;
    case 2:
      taskledsettings = CellRecalibratorSpec::LEDEventSettings::REDIRECT;
      break;
    default:
      LOG(fatal) << "Undefined handling of LED events";
  }
  std::vector<o2::framework::OutputSpec> outputs = {{o2::header::gDataOriginEMC, "CELLS", outputSubspec, o2::framework::Lifetime::Timeframe},
                                                    {o2::header::gDataOriginEMC, "CELLSTRGR", outputSubspec, o2::framework::Lifetime::Timeframe}};
  if (taskledsettings == CellRecalibratorSpec::LEDEventSettings::REDIRECT) {
    outputs.push_back({o2::header::gDataOriginEMC, "CELLS", 10, o2::framework::Lifetime::Timeframe});
    outputs.push_back({o2::header::gDataOriginEMC, "CELLSTRGR", 10, o2::framework::Lifetime::Timeframe});
  }
  calibhandler->defineInputSpecs(inputs);

  return o2::framework::DataProcessorSpec{"EMCALCellRecalibrator",
                                          inputs,
                                          outputs,
                                          o2::framework::adaptFromTask<o2::emcal::CellRecalibratorSpec>(outputSubspec, taskledsettings, badChannelCalib, timeCalib, gainCalib, calibhandler)};
}
