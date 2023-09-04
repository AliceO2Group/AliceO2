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

#include <iostream>
#include <vector>
#include <type_traits>
#include <gsl/span>

#include <fairlogger/Logger.h>

#include "Framework/ControlService.h"
#include "Framework/DataRefUtils.h"
#include "Framework/CCDBParamSpec.h"
#include "CommonConstants/Triggers.h"
#include "DataFormatsEMCAL/Cell.h"
#include "DataFormatsEMCAL/TriggerRecord.h"
#include "EMCALWorkflow/OfflineCalibSpec.h"

using namespace o2::emcal;

void OfflineCalibSpec::init(o2::framework::InitContext& ctx)
{
  // energy vs. cell ID
  mCellAmplitude = std::unique_ptr<TH2>(new TH2F("mCellAmplitude", "Cell amplitude", 800, 0., 40., 17664, -0.5, 17663.5));
  // time vs. cell ID
  mCellTime = std::unique_ptr<TH2>(new TH2F("mCellTime", "Cell time", 800, -200, 600, 17664, -0.5, 17663.5));
  // time vs. cell ID
  mCellTimeLG = std::unique_ptr<TH2>(new TH2F("mCellTimeLG", "Cell time (low gain)", 800, -200, 600, 17664, -0.5, 17663.5));
  // time vs. cell ID
  mCellTimeHG = std::unique_ptr<TH2>(new TH2F("mCellTimeHG", "Cell time (high gain)", 800, -200, 600, 17664, -0.5, 17663.5));
  // number of events
  mNevents = std::unique_ptr<TH1>(new TH1F("mNevents", "Number of events", 1, 0.5, 1.5));
  if (mMakeCellIDTimeEnergy) {
    // cell time, cell energy, cell ID
    std::array<int, 3> arrNBins = {17664, 100, 100}; // NCells, time, energy
    std::array<double, 3> arrMin = {-0.5, -50, 0};
    std::array<double, 3> arrMax = {17663.5, 50., 50};
    mCellTimeEnergy = std::unique_ptr<THnSparseF>(new THnSparseF("CellIDvsTimevsEnergy", "CellIDvsTimevsEnergy", arrNBins.size(), arrNBins.data(), arrMin.data(), arrMax.data()));
  }
}

void OfflineCalibSpec::finaliseCCDB(o2::framework::ConcreteDataMatcher& matcher, void* obj)
{
  LOG(info) << "Handling new Calibration objects";
  mCalibrationHandler->finalizeCCDB(matcher, obj);

  if (matcher == o2::framework::ConcreteDataMatcher("EMC", "EMCALCALIBPARAM", 0)) {
    LOG(info) << "EMCal CalibParams updated";
    EMCALCalibParams::Instance().printKeyValues(true, true);
  }

  if (mRejectL0Triggers && matcher == o2::framework::ConcreteDataMatcher("CTP", "CTPCONFIG", 0)) {
    // clear current class mask and prepare to fill in the updated values
    // The trigger names are seperated by a ":" in one string in the calib params
    mSelectedClassMasks.clear();
    std::string strSelClassMasks = EMCALCalibParams::Instance().selectedClassMasks;
    std::string delimiter = ":";
    size_t pos = 0;
    std::vector<std::string> vSelMasks;
    while ((pos = strSelClassMasks.find(delimiter)) != std::string::npos) {
      vSelMasks.push_back(strSelClassMasks.substr(0, pos));
      strSelClassMasks.erase(0, pos + delimiter.length());
    }
    vSelMasks.push_back(strSelClassMasks);

    auto ctpconf = reinterpret_cast<o2::ctp::CTPConfiguration*>(obj);

    for (auto& cls : ctpconf->getCTPClasses()) {
      LOG(debug) << "CTP class: " << cls.name << "\t " << cls.classMask;

      if (std::find(vSelMasks.begin(), vSelMasks.end(), cls.name) != vSelMasks.end()) {
        mSelectedClassMasks.push_back(cls.classMask);
        LOG(info) << "Setting selected class mask " << cls.name << " to bit " << cls.classMask;
      }
    }
  }
}

void OfflineCalibSpec::updateCalibObjects()
{
  if (mCalibrationHandler->hasUpdateGainCalib()) {
    LOG(info) << "updateCalibObjects: Gain calib params changed";
    mGainCalibFactors = mCalibrationHandler->getGainCalibration();
  }
}

void OfflineCalibSpec::run(framework::ProcessingContext& pc)
{
  auto cells = pc.inputs().get<gsl::span<o2::emcal::Cell>>("cells");
  auto triggerrecords = pc.inputs().get<gsl::span<o2::emcal::TriggerRecord>>("triggerrecord");

  // prepare CTPConfiguration such that it can be loaded in finalise ccdb
  if (mRejectL0Triggers) {
    pc.inputs().get<o2::ctp::CTPConfiguration*>(getCTPConfigBinding());
  }

  using ctpDigitsType = std::decay_t<decltype(pc.inputs().get<gsl::span<o2::ctp::CTPDigit>>(getCTPDigitsBinding()))>;
  std::optional<ctpDigitsType> ctpDigits;
  if (mRejectL0Triggers) {
    ctpDigits = pc.inputs().get<gsl::span<o2::ctp::CTPDigit>>(getCTPDigitsBinding());
  }

  mCalibrationHandler->checkUpdates(pc);
  updateCalibObjects();

  LOG(debug) << "[EMCALOfflineCalib - run] received " << cells.size() << " cells from " << triggerrecords.size() << " triggers ...";
  if (triggerrecords.size()) {
    for (const auto& trg : triggerrecords) {
      if (!trg.getNumberOfObjects()) {
        LOG(debug) << "[EMCALOfflineCalib - run] Trigger does not contain cells, skipping ...";
        continue;
      }

      // reject calibration triggers (EMCAL LED events etc.)
      if (mRejectCalibTriggers) {
        LOG(debug) << "Trigger: " << trg.getTriggerBits() << "   o2::trigger::Cal " << o2::trigger::Cal;
        if (trg.getTriggerBits() & o2::trigger::Cal) {
          LOG(debug) << "skipping triggered events due to wrong trigger (no Physics trigger)";
          continue;
        }
      }

      // reject all triggers that are not included in the classMask (typically only EMC min. bias should be accepted)
      uint64_t classMaskCTP = 0;
      if (mRejectL0Triggers) {
        bool acceptEvent = false;
        // Match the EMCal bc to the CTP bc
        int64_t bcEMC = trg.getBCData().toLong();
        for (auto& ctpDigit : *ctpDigits) {
          int64_t bcCTP = ctpDigit.intRecord.toLong();
          LOG(debug) << "bcEMC " << bcEMC << "   bcCTP " << bcCTP;
          if (bcCTP == bcEMC) {
            // obtain trigger mask that belongs to the selected bc
            classMaskCTP = ctpDigit.CTPClassMask.to_ulong();
            // now check if min bias trigger is not in mask
            for (const uint64_t& selectedClassMask : mSelectedClassMasks) {
              if ((classMaskCTP & selectedClassMask) != 0) {
                LOG(debug) << "trigger " << selectedClassMask << " found! accepting event";
                acceptEvent = true;
                break;
              }
            }
            break; // break as bc was matched
          }
        }
        // if current event is not accepted (selected triggers not present), move on to next event
        if (!acceptEvent) {
          continue;
        }
      }

      LOG(debug) << "[EMCALOfflineCalib - run] Trigger has " << trg.getNumberOfObjects() << " cells  ..." << std::endl;
      gsl::span<const o2::emcal::Cell> objectsTrigger(cells.data() + trg.getFirstEntry(), trg.getNumberOfObjects());
      for (const auto& c : objectsTrigger) {
        LOG(debug) << "[EMCALOfflineSpec - run] Channel: " << c.getTower();
        LOG(debug) << "[EMCALOfflineSpec - run] Energy: " << c.getEnergy();
        LOG(debug) << "[EMCALOfflineSpec - run] Time: " << c.getTimeStamp();
        LOG(debug) << "[EMCALOfflineSpec - run] IsLowGain: " << c.getLowGain();
        float cellE = c.getEnergy();
        if (mGainCalibFactors && mCalibrationHandler->hasGainCalib()) {
          LOG(debug) << "gain calib factor " << mGainCalibFactors->getGainCalibFactors(c.getTower());
          cellE *= mGainCalibFactors->getGainCalibFactors(c.getTower());
          LOG(debug) << "[EMCALOfflineSpec - run] corrected Energy: " << cellE;
        }
        mCellAmplitude->Fill(cellE, c.getTower());
        if (cellE > 0.5) {
          mCellTime->Fill(c.getTimeStamp(), c.getTower());
          if (c.getLowGain()) {
            mCellTimeLG->Fill(c.getTimeStamp(), c.getTower());
          } else { // high gain cells
            mCellTimeHG->Fill(c.getTimeStamp(), c.getTower());
          }
          if (mMakeCellIDTimeEnergy) {
            mCellTimeEnergy->Fill(c.getTower(), c.getTimeStamp(), cellE);
          }
        }
      }
      mNevents->Fill(1);
    }
  }
}

void OfflineCalibSpec::endOfStream(o2::framework::EndOfStreamContext& ec)
{
  // write histograms to root file here
  std::unique_ptr<TFile> outputFile(new TFile("emcal-offline-calib.root", "RECREATE"));
  outputFile->cd();
  mCellAmplitude->Write();
  mCellTime->Write();
  mCellTimeLG->Write();
  mCellTimeHG->Write();
  mNevents->Write();
  if (mMakeCellIDTimeEnergy) {
    mCellTimeEnergy->Write();
  }
  outputFile->Close();
}

o2::framework::DataProcessorSpec o2::emcal::getEmcalOfflineCalibSpec(bool makeCellIDTimeEnergy, bool rejectCalibTriggers, bool rejectL0Trigger, uint32_t inputsubspec, bool enableGainCalib, bool ctpcfgperrun)
{

  std::vector<o2::framework::InputSpec>
    inputs = {{"cells", o2::header::gDataOriginEMC, "CELLS", inputsubspec, o2::framework::Lifetime::Timeframe},
              {"triggerrecord", o2::header::gDataOriginEMC, "CELLSTRGR", inputsubspec, o2::framework::Lifetime::Timeframe}};

  if (rejectL0Trigger) {
    inputs.emplace_back(OfflineCalibSpec::getCTPConfigBinding(), "CTP", "CTPCONFIG", 0, o2::framework::Lifetime::Condition, o2::framework::ccdbParamSpec("CTP/Config/Config", ctpcfgperrun));
    inputs.emplace_back(OfflineCalibSpec::getCTPDigitsBinding(), "CTP", "DIGITS", 0, o2::framework::Lifetime::Timeframe);
  }
  auto calibhandler = std::make_shared<o2::emcal::CalibLoader>();
  calibhandler->enableGainCalib(enableGainCalib);
  calibhandler->defineInputSpecs(inputs);

  return o2::framework::DataProcessorSpec{"EMCALOfflineCalib",
                                          inputs,
                                          {},
                                          o2::framework::adaptFromTask<o2::emcal::OfflineCalibSpec>(makeCellIDTimeEnergy, rejectCalibTriggers, rejectL0Trigger, calibhandler)};
}
