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

/// \file   T0Fit.cxx
/// \brief  Fits the TRD PH distributions to extract the t0 value
/// \author Luisa Bergmann

#include "TRDCalibration/T0Fit.h"
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

#include "TMath.h"

#include <string>
#include <map>
#include <memory>
#include <ctime>

using namespace o2::trd::constants;

namespace o2::trd
{
//______________________________________________________________________________________________
double ErfLandauChi2Functor::operator()(const double* par) const
{
  // provides chi2 estimate of PH profile comparing the y-value of profiles to
  // par[0]*ROOT::Math::erf(x) + par[1]*TMath::Landau(x,par[2],par[3])
  //
  // par[0] : offset
  // par[1] : amplitude
  // par[2] : mean
  // par[3] : sigma

  double sum2 = 0;

  for (int i = lowerBoundFit; i <= upperBoundFit; ++i) {

    if (TMath::Abs(y[i]) < 1e-7) {
      continue;
    }

    double funcVal = par[0] * TMath::Erf(x[i]) + par[1] * TMath::Landau(x[i], par[2], par[3]);

    sum2 += TMath::Power(y[i] - funcVal, 2) / y[i];
  }

  return sum2;
}

//______________________________________________________________________________________________
using Slot = o2::calibration::TimeSlot<o2::trd::T0FitHistos>;

void T0Fit::initOutput()
{
  // reset the CCDB output vectors
  mInfoVector.clear();
  mObjectVector.clear();
}

void T0Fit::initProcessing()
{
  if (mInitDone) {
    return;
  }

  adcProfIncl = std::make_unique<TProfile>("adcProfIncl", "adcProfIncl", 30, -0.5, 29.5);
  for (int iDet = 0; iDet < constants::MAXCHAMBER; ++iDet) {
    adcProfDet[iDet] = std::make_unique<TProfile>(Form("adcProfDet_%i", iDet), Form("adcProfDet_%i", iDet), 30, -0.5, 29.5);
  }

  mFitFunctor.lowerBoundFit = 0;
  mFitFunctor.upperBoundFit = 5;

  double mParamsStart[4];
  mParamsStart[0] = 100;
  mParamsStart[1] = 500;
  mParamsStart[2] = 0.5;
  mParamsStart[3] = 0.5;

  mFitter.SetFCN<ErfLandauChi2Functor>(4, mFitFunctor, mParamsStart);

  ROOT::Math::MinimizerOptions opt;
  opt.SetMinimizerType("Minuit2");
  opt.SetMinimizerAlgorithm("Migrad");
  opt.SetPrintLevel(0);
  opt.SetMaxFunctionCalls(1000);
  opt.SetTolerance(.001);
  mFitter.Config().SetMinimizerOptions(opt);

  mFuncErfLandau = std::make_unique<TF1>(
    "fErfLandau", [&](double* x, double* par) { return par[0] * TMath::Erf(x[0]) + par[1] * TMath::Landau(x[0], par[2], par[3]); }, mFitFunctor.lowerBoundFit, mFitFunctor.upperBoundFit, 4);

  // set tree addresses
  if (mEnableOutput) {
    mOutTree->Branch("t0_chambers", &t0_chambers);
    mOutTree->Branch("t0_average", &t0_average);
  }

  mInitDone = true;
}

void T0Fit::finalizeSlot(Slot& slot)
{
  // do actual fits for the data provided in the given slot

  TStopwatch timer;
  timer.Start();
  initProcessing();

  // get data and fill profiles
  auto dataPH = slot.getContainer();

  for (int iEntry = 0; iEntry < dataPH->getNEntries(); ++iEntry) {
    int iDet = dataPH->getDetector(iEntry);
    adcProfIncl->Fill(dataPH->getTimeBin(iEntry), dataPH->getADC(iEntry));
    adcProfDet[iDet]->Fill(dataPH->getTimeBin(iEntry), dataPH->getADC(iEntry));
  }

  // do fits
  // inclusive distribution
  mFitFunctor.x.clear();
  mFitFunctor.y.clear();
  for (int i = 1; i <= 30; ++i) {
    mFitFunctor.x.push_back(i - 1);
    mFitFunctor.y.push_back(adcProfIncl->GetBinContent(i));
  }

  mFitter.FitFCN();
  auto fitResult = mFitter.Result();

  if (fitResult.IsValid()) {
    mFuncErfLandau->SetParameters(fitResult.GetParams()[0], fitResult.GetParams()[1], fitResult.GetParams()[2], fitResult.GetParams()[3]);
    t0_average = mFuncErfLandau->GetMaximumX();
  } else {
    LOG(warn) << "t0 fit for inclusive distribtion is not valid, set to " << mDummyT0;
    t0_average = mDummyT0;
  }

  // single chambers
  for (int iDet = 0; iDet < constants::MAXCHAMBER; ++iDet) {
    if (adcProfDet[iDet]->GetEntries() < mParams.minEntriesChamberT0Fit) {
      LOG(info) << "not enough entries in chamber " << iDet << " for t0 fit, set to " << mDummyT0;
      t0_chambers[iDet] = mDummyT0;
      continue;
    }

    mFitFunctor.x.clear();
    mFitFunctor.y.clear();
    for (int i = 1; i <= 30; ++i) {
      mFitFunctor.x.push_back(i - 1);
      mFitFunctor.y.push_back(adcProfDet[iDet]->GetBinContent(i));
    }

    mFitter.FitFCN();
    fitResult = mFitter.Result();

    if (fitResult.IsValid()) {
      mFuncErfLandau->SetParameters(fitResult.GetParams()[0], fitResult.GetParams()[1], fitResult.GetParams()[2], fitResult.GetParams()[3]);
      t0_chambers[iDet] = mFuncErfLandau->GetMaximumX();
    } else {
      LOG(info) << "t0 fit for chamber " << iDet << " is not valid, set to " << mDummyT0;
      t0_chambers[iDet] = mDummyT0;
    }
  }

  // fill tree
  if (mEnableOutput) {
    mOutTree->Fill();

    LOGF(debug, "Fit result for inclusive distribution: t0 = ", t0_average);
    for (int iDet = 0; iDet < MAXCHAMBER; ++iDet) {
      LOGF(debug, "Fit result for chamber %i: t0 = ", iDet, t0_chambers[iDet]);
    }
  }

  // assemble CCDB object
  CalT0 t0Object;
  t0Object.setT0av(t0_average);
  for (int iDet = 0; iDet < constants::MAXCHAMBER; ++iDet) {
    t0Object.setT0(iDet, t0_chambers[iDet]);
  }
  auto clName = o2::utils::MemFileHelper::getClassName(t0Object);
  auto flName = o2::ccdb::CcdbApi::generateFileName(clName);
  std::map<std::string, std::string> metadata; // TODO: do we want to store any meta data?
  long startValidity = slot.getStartTimeMS();
  mInfoVector.emplace_back("TRD/Calib/CalT0", clName, flName, metadata, startValidity, startValidity + o2::ccdb::CcdbObjectInfo::HOUR);
  mObjectVector.push_back(t0Object);
}

Slot& T0Fit::emplaceNewSlot(bool front, TFType tStart, TFType tEnd)
{
  auto& container = getSlots();
  auto& slot = front ? container.emplace_front(tStart, tEnd) : container.emplace_back(tStart, tEnd);
  slot.setContainer(std::make_unique<T0FitHistos>());
  return slot;
}

void T0Fit::createOutputFile()
{
  mEnableOutput = true;
  mOutFile = std::make_unique<TFile>("trd_calt0.root", "RECREATE");
  if (mOutFile->IsZombie()) {
    LOG(error) << "Failed to create output file!";
    mEnableOutput = false;
    return;
  }
  mOutTree = std::make_unique<TTree>("calib", "t0 values");
  LOG(info) << "Created output file trd_calt0.root";
}

void T0Fit::closeOutputFile()
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
    LOG(error) << "Failed to write t0-value data file, reason: " << e.what();
  }
  mEnableOutput = false;
}
} // namespace o2::trd
