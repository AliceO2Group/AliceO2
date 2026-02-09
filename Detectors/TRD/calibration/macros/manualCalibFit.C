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

/// \file manualCalibFit.C
/// \author Felix Schlepper

#if !defined(__CLING__) || defined(__ROOTCLING__)

#include <array>
#include <memory>
#include <cmath>
#include <utility>
#include <vector>

// ROOT header
#include <TBranch.h>
#include <TCanvas.h>
#include <TChain.h>
#include <TProfile.h>
#include <TFile.h>
#include <Fit/Fitter.h>

// O2 header
#include <TRDCalibration/CalibratorVdExB.h>
#include "DetectorsBase/Propagator.h"

#endif

// This root macro reads in 'trdangreshistos.root' and
// performs the calibration fits manually as in CalibratorVdExB.cxx
// This can be used for checking if the calibration fits make sense.
void manualCalibFit(int runNumber = 563335, bool usePreCorrFromCCDB = false)
{
  //----------------------------------------------------
  // TTree and File
  //----------------------------------------------------
  std::unique_ptr<TFile> inFilePtr(TFile::Open("trdcaliboutput.root"));
  if (inFilePtr == nullptr) {
    printf("Input File could not be read!\n'");
    return;
  }
  auto tree = inFilePtr->Get<TTree>("calibdata");
  if (tree == nullptr) {
    printf("Tree 'calibdata' not in file!\n");
    return;
  }
  Float_t mHistogramEntries[13500];
  Int_t mNEntriesPerBin[13500];
  std::array<Float_t, 13500> mHistogramEntriesSum;
  std::array<Int_t, 13500> mNEntriesPerBinSum;
  mHistogramEntriesSum.fill(0.f);
  mNEntriesPerBinSum.fill(0);
  tree->SetBranchAddress("mHistogramEntries[13500]", &mHistogramEntries);
  tree->SetBranchAddress("mNEntriesPerBin[13500]", &mNEntriesPerBin);

  // use precorr values from ccdb
  // necessary when the angular residuals were calculated already using ccdb calibration (e.g. in a local run)

  o2::trd::CalVdriftExB* calObject;
  if (usePreCorrFromCCDB) {
    auto& ccdbmgr = o2::ccdb::BasicCCDBManager::instance();

    o2::ccdb::CcdbApi ccdb;
    ccdb.init("http://alice-ccdb.cern.ch");
    auto runDuration = ccdbmgr.getRunDuration(runNumber);

    std::map<std::string, std::string> metadata;
    std::map<std::string, std::string> headers;

    calObject = ccdb.retrieveFromTFileAny<o2::trd::CalVdriftExB>("TRD/Calib/CalVdriftExB", metadata, runDuration.first + 60000, &headers, "", "", "1689478811721");
  }

  //----------------------------------------------------
  // Configure Fitter
  //----------------------------------------------------
  o2::trd::FitFunctor mFitFunctor;
  std::array<std::unique_ptr<TProfile>, 540> profiles; ///< profile histograms for each TRD chamber
  int counter = 0;
  for (int iDet = 0; iDet < 540; ++iDet) {
    mFitFunctor.profiles[iDet] = std::make_unique<TProfile>(Form("profAngleDiff_%i", iDet), Form("profAngleDiff_%i", iDet), 25, -25.f, 25.f);
    if (usePreCorrFromCCDB) {
      if (calObject->isGoodExB(iDet))
        counter++;
      mFitFunctor.vdPreCorr[iDet] = calObject->getVdrift(iDet, true);
      mFitFunctor.laPreCorr[iDet] = calObject->getExB(iDet, true);
    }
  }
  std::cout << counter << " good entries in the CCDB " << std::endl;
  mFitFunctor.mAnodePlane = 3.35; // don't really care as long as it's not zero, this parameter could  be removed
  mFitFunctor.lowerBoundAngleFit = 80 * TMath::DegToRad();
  mFitFunctor.upperBoundAngleFit = 100 * TMath::DegToRad();
  if (!usePreCorrFromCCDB) {
    mFitFunctor.vdPreCorr.fill(1.546);
    mFitFunctor.laPreCorr.fill(0.0);
  }

  //----------------------------------------------------
  // Loop
  //----------------------------------------------------
  for (Int_t iEntry = 0; tree->LoadTree(iEntry) >= 0; ++iEntry) {
    // Load data
    tree->GetEntry(iEntry);
    for (int iBin = 0; iBin < 13500; ++iBin) { // Sum the histograms from different tfs
      mHistogramEntriesSum[iBin] += mHistogramEntries[iBin];
      mNEntriesPerBinSum[iBin] += mNEntriesPerBin[iBin];
    }
  }

  //----------------------------------------------------
  // Fill profiles
  //----------------------------------------------------
  int nEntriesDetTotal[540] = {};
  for (int iDet = 0; iDet < 540; ++iDet) {
    for (int iBin = 0; iBin < 25; ++iBin) {
      auto angleDiffSum = mHistogramEntriesSum[iDet * 25 + iBin];
      auto nEntries = mNEntriesPerBinSum[iDet * 25 + iBin];
      nEntriesDetTotal[iDet] += nEntries;
      if (nEntries > 0) { // skip entries which have no entries; ?
        // add to the respective profile for fitting later on
        mFitFunctor.profiles[iDet]->Fill(2 * iBin - 25.f, angleDiffSum / nEntries, nEntries);
      }
    }
    printf("Det %d: nEntries=%d \n", iDet, nEntriesDetTotal[iDet]);
  }

  //----------------------------------------------------
  // Fitting
  //----------------------------------------------------
  printf("-------- Started fits\n");
  std::array<float, 540> laFitResults{};
  std::array<float, 540> vdFitResults{};

  TH1F* hVd = new TH1F("hVd", "v drift", 150, 0.5, 2.);
  TH1F* hLa = new TH1F("hLa", "lorentz angle", 200, -25., 25.);
  o2::trd::CalVdriftExB* calObjectOut = new o2::trd::CalVdriftExB();

  for (int iDet = 0; iDet < 540; ++iDet) {
    if (nEntriesDetTotal[iDet] < 75)
      continue;
    mFitFunctor.currDet = iDet;
    ROOT::Fit::Fitter fitter;
    double paramsStart[2];
    paramsStart[0] = 0.;
    paramsStart[1] = 1.;
    fitter.SetFCN<o2::trd::FitFunctor>(2, mFitFunctor, paramsStart);
    fitter.Config().ParSettings(0).SetLimits(-0.7, 0.7);
    fitter.Config().ParSettings(0).SetStepSize(.01);
    fitter.Config().ParSettings(1).SetLimits(0.01, 3.);
    fitter.Config().ParSettings(1).SetStepSize(.01);
    ROOT::Math::MinimizerOptions opt;
    opt.SetMinimizerType("Minuit2");
    opt.SetMinimizerAlgorithm("Migrad");
    opt.SetPrintLevel(0);
    opt.SetMaxFunctionCalls(1'000);
    opt.SetTolerance(.001);
    fitter.Config().SetMinimizerOptions(opt);
    fitter.FitFCN();
    auto fitResult = fitter.Result();
    laFitResults[iDet] = fitResult.Parameter(0);
    vdFitResults[iDet] = fitResult.Parameter(1);
    if (fitResult.MinFcnValue() > 0.03)
      continue;
    printf("Det %d: la=%.3f \tvd=%.3f \t100*minValue=%f \tentries=%d\n", iDet, laFitResults[iDet] * TMath::RadToDeg(), vdFitResults[iDet], 100 * fitResult.MinFcnValue(), nEntriesDetTotal[iDet]);
    hVd->Fill(vdFitResults[iDet]);
    hLa->Fill(laFitResults[iDet] * TMath::RadToDeg());
    calObjectOut->setVdrift(iDet, vdFitResults[iDet]);
    calObjectOut->setExB(iDet, laFitResults[iDet]);
  }
  printf("-------- Finished fits\n");

  std::cout << "number of chambers with enough entries: " << hVd->GetEntries() << std::endl;
  ;
  std::cout << "vdrift mean: " << hVd->GetMean() << " sigma: " << hVd->GetStdDev() << std::endl;
  std::cout << "lorentz angle mean: " << hLa->GetMean() << " sigma: " << hLa->GetStdDev() << std::endl;

  //----------------------------------------------------
  // Write
  //----------------------------------------------------
  std::unique_ptr<TFile> outFilePtr(TFile::Open("manualCalibFit.root", "RECREATE"));
  hVd->Write();
  hLa->Write();
  outFilePtr->WriteObjectAny(calObjectOut, "o2::trd::CalVdriftExB", "calObject");
  for (auto& p : mFitFunctor.profiles)
    p->Write();
}
