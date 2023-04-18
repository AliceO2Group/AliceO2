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

/// \file CalibratorVdExB.cxx
/// \brief TimeSlot-based calibration of vDrift and ExB
/// \author Ole Schmidt

#include "TRDCalibration/CalibratorVdExB.h"
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

#include <string>
#include <map>
#include <memory>
#include <ctime>

using namespace o2::trd::constants;

namespace o2::trd
{

double FitFunctor::calculateDeltaAlphaSim(double vdFit, double laFit, double impactAng) const
{
  auto xDir = TMath::Cos(impactAng);
  auto yDir = TMath::Sin(impactAng);
  double slope = (TMath::Abs(xDir) < 1e-7) ? 1e7 : yDir / xDir;
  auto laTan = TMath::Tan(laFit);
  double lorentzSlope = (TMath::Abs(laTan) < 1e-7) ? 1e7 : 1. / laTan;

  // hit point of incoming track with anode plane
  double xAnodeHit = mAnodePlane / slope;
  double yAnodeHit = mAnodePlane;

  // hit point at anode plane of Lorentz angle shifted cluster from the entrance -> independent of true drift velocity
  double xLorentzAnodeHit = mAnodePlane / lorentzSlope;
  double yLorentzAnodeHit = mAnodePlane;

  // cluster location within drift cell of cluster from entrance after drift velocity ratio is applied
  double xLorentzDriftHit = xLorentzAnodeHit;
  double yLorentzDriftHit = mAnodePlane - mAnodePlane * (vdPreCorr[currDet] / vdFit);

  // reconstructed hit of first cluster at chamber entrance after pre Lorentz angle correction
  double xLorentzDriftHitPreCorr = xLorentzAnodeHit - (mAnodePlane - yLorentzDriftHit) * TMath::Tan(laPreCorr[currDet]);
  double yLorentzDriftHitPreCorr = mAnodePlane - mAnodePlane * (vdPreCorr[currDet] / vdFit);

  double impactAngleSim = TMath::ATan2(yAnodeHit, xAnodeHit);

  double deltaXLorentzDriftHit = xAnodeHit - xLorentzDriftHitPreCorr;
  double deltaYLorentzDriftHit = yAnodeHit - yLorentzDriftHitPreCorr;
  double impactAngleRec = TMath::ATan2(deltaYLorentzDriftHit, deltaXLorentzDriftHit);

  double deltaAngle = (impactAngleRec - impactAngleSim); // * TMath::RadToDeg());

  return deltaAngle;
}

double FitFunctor::operator()(const double* par) const
{
  double sum = 0; // this value is minimized
  for (int iBin = 1; iBin <= profiles[currDet]->GetNbinsX(); ++iBin) {
    auto impactAngle = (profiles[currDet]->GetBinCenter(iBin) + 90) * TMath::DegToRad();
    auto deltaAlpha = profiles[currDet]->GetBinContent(iBin) * TMath::DegToRad();
    if (TMath::Abs(deltaAlpha) < 1e-7) {
      continue;
    }
    if (impactAngle < lowerBoundAngleFit || impactAngle > upperBoundAngleFit) {
      continue;
    }
    auto deltaAlphaSim = calculateDeltaAlphaSim(par[CalibratorVdExB::ParamIndex::VD], par[CalibratorVdExB::ParamIndex::LA], impactAngle);
    sum += TMath::Power(deltaAlphaSim - deltaAlpha, 2);
  }
  return sum;
}

using Slot = o2::calibration::TimeSlot<AngularResidHistos>;

void CalibratorVdExB::initOutput()
{
  // reset the CCDB output vectors
  mInfoVector.clear();
  mObjectVector.clear();
}

void CalibratorVdExB::initProcessing()
{
  if (mInitDone) {
    return;
  }

  mFitFunctor.lowerBoundAngleFit = 80 * TMath::DegToRad();
  mFitFunctor.upperBoundAngleFit = 100 * TMath::DegToRad();
  mFitFunctor.mAnodePlane = GeometryBase::camHght() / (2.f * 100.f);
  for (int iDet = 0; iDet < MAXCHAMBER; ++iDet) {
    mFitFunctor.profiles[iDet] = std::make_unique<TProfile>(Form("profAngleDiff_%i", iDet), Form("profAngleDiff_%i", iDet), NBINSANGLEDIFF, -MAXIMPACTANGLE, MAXIMPACTANGLE);
  }

  mFitter.SetFCN<FitFunctor>(2, mFitFunctor, mParamsStart);
  mFitter.Config().ParSettings(ParamIndex::LA).SetLimits(-0.7, 0.7);
  mFitter.Config().ParSettings(ParamIndex::LA).SetStepSize(.01);
  mFitter.Config().ParSettings(ParamIndex::VD).SetLimits(0.01, 3.);
  mFitter.Config().ParSettings(ParamIndex::VD).SetStepSize(.01);
  ROOT::Math::MinimizerOptions opt;
  opt.SetMinimizerType("Minuit2");
  opt.SetMinimizerAlgorithm("Migrad");
  opt.SetPrintLevel(0);
  opt.SetMaxFunctionCalls(1'000);
  opt.SetTolerance(.001);
  mFitter.Config().SetMinimizerOptions(opt);

  // set tree addresses
  if (mEnableOutput && mOutTree) {
    mOutTree->Branch("lorentzAngle", &mFitFunctor.laPreCorr);
    mOutTree->Branch("vDrift", &mFitFunctor.vdPreCorr);
    for (int iDet = 0; iDet < MAXCHAMBER; ++iDet) {
      mOutTree->Branch(fmt::format("residuals_{:d}", iDet).c_str(), mFitFunctor.profiles[iDet].get());
    }
  }

  mInitDone = true;
}

void CalibratorVdExB::retrievePrev(o2::framework::ProcessingContext& pc)
{
  static bool doneOnce = false;
  if (!doneOnce) {
    doneOnce = true;
    mFitFunctor.vdPreCorr.fill(constants::VDRIFTDEFAULT);
    mFitFunctor.laPreCorr.fill(constants::EXBDEFAULT);
    // We either get a pointer to a valid object from the last ~hour or to the default object
    // which is always present. The first has precedence over the latter.
    auto dataCalVdriftExB = pc.inputs().get<o2::trd::CalVdriftExB*>("calvdexb");
    std::string msg = "Default Object";
    // We check if the object we got is the default one by comparing it to the defaults.
    for (int iDet = 0; iDet < MAXCHAMBER; ++iDet) {
      if (dataCalVdriftExB->getVdrift(iDet) != constants::VDRIFTDEFAULT ||
          dataCalVdriftExB->getExB(iDet) != constants::EXBDEFAULT) {
        msg = "Previous Object";
        break;
      }
    }
    LOG(info) << "Calibrator: From CCDB retrieved " << msg;

    // Here we set each entry regardless if it is the default or not.
    for (int iDet = 0; iDet < MAXCHAMBER; ++iDet) {
      mFitFunctor.laPreCorr[iDet] = dataCalVdriftExB->getExB(iDet);
      mFitFunctor.vdPreCorr[iDet] = dataCalVdriftExB->getVdrift(iDet);
    }
  }
}

void CalibratorVdExB::finalizeSlot(Slot& slot)
{
  // do actual calibration for the data provided in the given slot
  TStopwatch timer;
  timer.Start();
  initProcessing();
  auto laFitResults = mFitFunctor.laPreCorr;
  auto vdFitResults = mFitFunctor.vdPreCorr;
  auto residHists = slot.getContainer();
  for (int iDet = 0; iDet < MAXCHAMBER; ++iDet) {
    int sumEntries = 0;
    mFitFunctor.profiles[iDet]->Reset();
    mFitFunctor.currDet = iDet;
    for (int iBin = 0; iBin < NBINSANGLEDIFF; ++iBin) {
      // fill profiles
      auto angleDiffSum = residHists->getHistogramEntry(iDet * NBINSANGLEDIFF + iBin);
      auto nEntries = residHists->getBinCount(iDet * NBINSANGLEDIFF + iBin);
      sumEntries += nEntries;
      if (nEntries > 0) {
        mFitFunctor.profiles[iDet]->Fill(2 * iBin - MAXIMPACTANGLE, angleDiffSum / nEntries, nEntries);
      }
    }
    // Check if we have the minimum amount of entries
    if (sumEntries < mMinEntriesChamber) {
      LOGF(debug, "Chamber %d did not reach minimum amount of entries for refit", iDet);
      continue;
    }
    // Reset Start Parameter
    mParamsStart[ParamIndex::LA] = 0.0;
    mParamsStart[ParamIndex::VD] = 1.0;
    mFitter.FitFCN();
    auto fitResult = mFitter.Result();
    laFitResults[iDet] = fitResult.Parameter(ParamIndex::LA);
    vdFitResults[iDet] = fitResult.Parameter(ParamIndex::VD);
    LOGF(debug, "Fit result for chamber %i: vd=%f, la=%f", iDet, vdFitResults[iDet], laFitResults[iDet] * TMath::RadToDeg());
    // Update fit values for next fit
    mFitFunctor.laPreCorr[iDet] = laFitResults[iDet];
    mFitFunctor.vdPreCorr[iDet] = vdFitResults[iDet];
  }
  timer.Stop();
  LOGF(info, "Done fitting angular residual histograms. CPU time: %f, real time: %f", timer.CpuTime(), timer.RealTime());

  // Fill Tree and log to debug
  if (mEnableOutput && mOutTree) {
    mOutTree->Fill();
    for (int iDet = 0; iDet < MAXCHAMBER; ++iDet) {
      LOGF(debug, "Fit result for chamber %i: vd=%f, la=%f", iDet, vdFitResults[iDet], laFitResults[iDet] * TMath::RadToDeg());
    }
  }

  // assemble CCDB object
  CalVdriftExB calObject;
  for (int iDet = 0; iDet < MAXCHAMBER; ++iDet) {
    // For Chambers which did not have the minimum amount of entries in this slot e.g. missing, empty chambers.
    // We will reuse the prevoius one. This would have been read either from the ccdb or come from a previous successful fit.
    calObject.setVdrift(iDet, vdFitResults[iDet]);
    calObject.setExB(iDet, laFitResults[iDet]);
  }
  auto clName = o2::utils::MemFileHelper::getClassName(calObject);
  auto flName = o2::ccdb::CcdbApi::generateFileName(clName);
  std::map<std::string, std::string> metadata; // TODO: do we want to store any meta data?
  long startValidity = slot.getStartTimeMS() - 10 * o2::ccdb::CcdbObjectInfo::SECOND;
  mInfoVector.emplace_back("TRD/Calib/CalVdriftExB", clName, flName, metadata, startValidity, startValidity + o2::ccdb::CcdbObjectInfo::HOUR);
  mObjectVector.push_back(calObject);
}

Slot& CalibratorVdExB::emplaceNewSlot(bool front, TFType tStart, TFType tEnd)
{
  auto& container = getSlots();
  auto& slot = front ? container.emplace_front(tStart, tEnd) : container.emplace_back(tStart, tEnd);
  slot.setContainer(std::make_unique<AngularResidHistos>());
  return slot;
}

void CalibratorVdExB::createOutputFile()
{
  if (!mEnableOutput) {
    return;
  }

  mOutFile = std::make_unique<TFile>("trd_calibVdriftExB.root", "RECREATE");
  if (mOutFile->IsZombie()) {
    LOG(error) << "Failed to create output file!";
    mEnableOutput = false;
    return;
  }

  mOutTree = std::make_unique<TTree>("calib", "VDrift&ExB calibration");

  LOG(info) << "Created output file";
}

void CalibratorVdExB::closeOutputFile()
{
  if (!mEnableOutput || !mOutTree) {
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
}

} // namespace o2::trd
