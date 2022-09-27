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

#include "FairLogger.h"

#include "Framework/ControlService.h"
#include "Framework/DataRefUtils.h"
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

void OfflineCalibSpec::run(framework::ProcessingContext& pc)
{
  auto cells = pc.inputs().get<gsl::span<o2::emcal::Cell>>("cells");
  auto triggerrecords = pc.inputs().get<gsl::span<o2::emcal::TriggerRecord>>("triggerrecord");
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

      LOG(debug) << "[EMCALOfflineCalib - run] Trigger has " << trg.getNumberOfObjects() << " cells  ..." << std::endl;
      gsl::span<const o2::emcal::Cell> objectsTrigger(cells.data() + trg.getFirstEntry(), trg.getNumberOfObjects());
      for (const auto& c : objectsTrigger) {
        LOG(debug) << "[EMCALOfflineSpec - run] Channel: " << c.getTower();
        LOG(debug) << "[EMCALOfflineSpec - run] Energy: " << c.getEnergy();
        LOG(debug) << "[EMCALOfflineSpec - run] Time: " << c.getTimeStamp();
        LOG(debug) << "[EMCALOfflineSpec - run] IsLowGain: " << c.getLowGain();
        mCellAmplitude->Fill(c.getEnergy(), c.getTower());
        if (c.getEnergy() > 0.5) {
          mCellTime->Fill(c.getTimeStamp(), c.getTower());
          if (c.getLowGain()) {
            mCellTimeLG->Fill(c.getTimeStamp(), c.getTower());
          } else { // high gain cells
            mCellTimeHG->Fill(c.getTimeStamp(), c.getTower());
          }
          if (mMakeCellIDTimeEnergy) {
            mCellTimeEnergy->Fill(c.getTower(), c.getTimeStamp(), c.getEnergy());
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

o2::framework::DataProcessorSpec o2::emcal::getEmcalOfflineCalibSpec(bool makeCellIDTimeEnergy, bool rejectCalibTriggers)
{
  return o2::framework::DataProcessorSpec{"EMCALOfflineCalib",
                                          {{"cells", o2::header::gDataOriginEMC, "CELLS", 0, o2::framework::Lifetime::Timeframe},
                                           {"triggerrecord", o2::header::gDataOriginEMC, "CELLSTRGR", 0, o2::framework::Lifetime::Timeframe}},
                                          {},
                                          o2::framework::adaptFromTask<o2::emcal::OfflineCalibSpec>(makeCellIDTimeEnergy, rejectCalibTriggers)};
}
