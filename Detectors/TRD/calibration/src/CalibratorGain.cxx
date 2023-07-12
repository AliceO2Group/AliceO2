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

/// \file CalibratorGain.cxx
/// \brief TimeSlot-based calibration of gain
/// \author Gauthier Legras

#include "TRDCalibration/CalibratorGain.h"
#include "Framework/ProcessingContext.h"
#include "Framework/TimingInfo.h"
#include "Framework/InputRecord.h"
#include "DataFormatsTRD/Constants.h"
#include "TRDBase/GeometryBase.h"
#include "TStopwatch.h"
#include "CCDB/CcdbApi.h"
#include "CCDB/BasicCCDBManager.h"
#include "CommonUtils/NameConf.h"
#include "CommonUtils/MemFileHelper.h"
#include <TFile.h>
#include <TTree.h>
#include <TF1Convolution.h>

#include <string>
#include <map>
#include <memory>
#include <ctime>

using namespace o2::trd::constants;

namespace o2::trd
{

using Slot = o2::calibration::TimeSlot<GainCalibHistos>;

void CalibratorGain::initOutput()
{
  // reset the CCDB output vectors
  mInfoVector.clear();
  mObjectVector.clear();
}

void CalibratorGain::initProcessing()
{
  if (mInitDone) {
    return;
  }
  for (int iDet = 0; iDet < MAXCHAMBER; ++iDet) {
    mdEdxhists[iDet] = std::make_unique<TH1F>(Form("hdEdx%d", iDet), "dEdx", NBINSGAINCALIB, 0., NBINSGAINCALIB);
  }

  // Init fitting functions
  mFconv = std::make_unique<TF1Convolution>("[3] * TMath::Landau(x, [0], [1]) * exp(-[2]*x)", "TMath::Gaus(x, 0, [0])", 0., NBINSGAINCALIB, true);
  mFconv->SetNofPointsFFT(2000);
  mFitFunction = std::make_unique<TF1>("fitConvLandau", *mFconv, 0., NBINSGAINCALIB, mFconv->GetNpar());
  mFitFunction->SetParameters(40, 15, 0.02, 1, 0.1);
  mFitFunction->SetParLimits(0, 30., 100.0);
  mFitFunction->SetParLimits(1, 5.0, 25.0);
  mFitFunction->SetParLimits(2, -0.1, 0.5);
  mFitFunction->SetParLimits(3, 1.0, 10.0);
  mFitFunction->SetParLimits(4, -0.1, 0.5);

  // set tree addresses
  if (mEnableOutput) {
    mOutTree->Branch("MPVdEdx", &mFitResults);
    for (int iDet = 0; iDet < MAXCHAMBER; ++iDet) {
      mOutTree->Branch(Form("dEdxHist_%d", iDet), mdEdxhists[iDet].get());
    }
  }

  mInitDone = true;
}

void CalibratorGain::retrievePrev(o2::framework::ProcessingContext& pc)
{

  mFitResults.fill(constants::MPVDEDXDEFAULT);
  // We either get a pointer to a valid object from the last ~hour or to the default object
  // which is always present. The first has precedence over the latter.
  auto dataCalGain = pc.inputs().get<o2::trd::CalGain*>("calgain");
  std::string msg = "Default Object";
  // We check if the object we got is the default one by comparing it to the defaults.
  for (int iDet = 0; iDet < MAXCHAMBER; ++iDet) {
    if (dataCalGain->getMPVdEdx(iDet) != constants::MPVDEDXDEFAULT) {
      msg = "Previous Object";
      break;
    }
  }
  LOG(info) << "Calibrator: From CCDB retrieved " << msg;

  // Here we set each entry regardless if it is the default or not.
  for (int iDet = 0; iDet < MAXCHAMBER; ++iDet) {
    mFitResults[iDet] = dataCalGain->getMPVdEdx(iDet);
  }
}

void CalibratorGain::finalizeSlot(Slot& slot)
{
  // do actual calibration for the data provided in the given slot
  TStopwatch timer;
  timer.Start();
  initProcessing();

  auto dEdxHists = slot.getContainer();
  for (int iDet = 0; iDet < MAXCHAMBER; ++iDet) {
    mdEdxhists[iDet]->Reset();
    for (int iBin = 0; iBin < NBINSGAINCALIB; ++iBin) {
      mdEdxhists[iDet]->SetBinContent(iBin + 1, dEdxHists->getHistogramEntry(iDet * NBINSGAINCALIB + iBin));
      mdEdxhists[iDet]->SetBinError(iBin + 1, sqrt(dEdxHists->getHistogramEntry(iDet * NBINSGAINCALIB + iBin)));
    }
    int nEntries = mdEdxhists[iDet]->Integral();
    // Check if we have the minimum amount of entries
    if (nEntries < mMinEntriesChamber) {
      LOGF(debug, "Chamber %d did not reach minimum amount of %d entries for refit", iDet, mMinEntriesChamber);
      continue;
    }
    mdEdxhists[iDet]->Scale(1. / nEntries);

    // Fitting histogram
    mFitFunction->SetParameter(0, mdEdxhists[iDet]->GetMean() / 1.25);
    int fitStatus = mdEdxhists[iDet]->Fit("fitConvLandau", "LQB", "", 1, NBINSGAINCALIB - 4);

    if (fitStatus != 0) {
      LOGF(warn, "Fit for chamber %i failed, nEntries: %d", iDet, nEntries);
      continue;
    }

    mFitResults[iDet] = mFitFunction->GetMaximumX(0., NBINSGAINCALIB);
    LOGF(debug, "Fit result for chamber %i: dEdx MPV = ", iDet, mFitResults[iDet]);
  }
  timer.Stop();
  LOGF(info, "Done fitting dEdx histograms. CPU time: %f, real time: %f", timer.CpuTime(), timer.RealTime());

  // Fill Tree and log to debug
  if (mEnableOutput) {
    mOutTree->Fill();
    for (int iDet = 0; iDet < MAXCHAMBER; ++iDet) {
      LOGF(debug, "Fit result for chamber %i: dEdx MPV = ", iDet, mFitResults[iDet]);
    }
  }

  // assemble CCDB object
  CalGain calObject;
  for (int iDet = 0; iDet < MAXCHAMBER; ++iDet) {
    // For Chambers which did not have the minimum amount of entries in this slot e.g. missing, empty chambers.
    // We will reuse the previous one. This would have been read either from the ccdb or come from a previous successful fit.
    calObject.setMPVdEdx(iDet, mFitResults[iDet]);
  }
  auto clName = o2::utils::MemFileHelper::getClassName(calObject);
  auto flName = o2::ccdb::CcdbApi::generateFileName(clName);
  std::map<std::string, std::string> metadata; // TODO: do we want to store any meta data?
  long startValidity = slot.getStartTimeMS();
  mInfoVector.emplace_back("TRD/Calib/CalGain", clName, flName, metadata, startValidity, startValidity + 3 * o2::ccdb::CcdbObjectInfo::DAY);
  mObjectVector.push_back(calObject);
}

Slot& CalibratorGain::emplaceNewSlot(bool front, TFType tStart, TFType tEnd)
{
  auto& container = getSlots();
  auto& slot = front ? container.emplace_front(tStart, tEnd) : container.emplace_back(tStart, tEnd);
  slot.setContainer(std::make_unique<GainCalibHistos>());
  return slot;
}

void CalibratorGain::createOutputFile()
{
  mEnableOutput = true;
  mOutFile = std::make_unique<TFile>("trd_calibgain.root", "RECREATE");
  if (mOutFile->IsZombie()) {
    LOG(error) << "Failed to create output file!";
    mEnableOutput = false;
    return;
  }

  mOutTree = std::make_unique<TTree>("calib", "Gain calibration");

  LOG(info) << "Created output file trd_calibgain.root";
}

void CalibratorGain::closeOutputFile()
{
  if (!mEnableOutput) {
    return;
  }

  try {
    mOutFile->cd();
    mOutTree->Write();
    mOutTree.reset();
    mOutFile->Close();
    mOutFile.reset();
  } catch (std::exception const& e) {
    LOG(error) << "Failed to write calibration data file, reason: " << e.what();
  }
  mEnableOutput = false;
}

} // namespace o2::trd
