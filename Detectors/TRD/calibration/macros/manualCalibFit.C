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

#endif

// This root macro reads in 'trdangreshistos.root' and
// performs the calibration fits manually as in CalibratorVdExB.cxx
// This can be used for checking if the calibration fits make sense.
void manualCalibFit()
{
  //----------------------------------------------------
  // TTree and File
  //----------------------------------------------------
  std::unique_ptr<TFile> inFilePtr(TFile::Open("trdangreshistos.root"));
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

  //----------------------------------------------------
  // Configure Fitter
  //----------------------------------------------------
  o2::trd::FitFunctor mFitFunctor;
  std::array<std::unique_ptr<TProfile>, 540> profiles; ///< profile histograms for each TRD chamber
  for (int iDet = 0; iDet < 540; ++iDet) {
    mFitFunctor.profiles[iDet] = std::make_unique<TProfile>(Form("profAngleDiff_%i", iDet), Form("profAngleDiff_%i", iDet), 25, -25.f, 25.f);
  }
  mFitFunctor.lowerBoundAngleFit = 80 * TMath::DegToRad();
  mFitFunctor.upperBoundAngleFit = 100 * TMath::DegToRad();
  mFitFunctor.vdPreCorr.fill(1.546);
  mFitFunctor.laPreCorr.fill(0.0);

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
  for (int iDet = 0; iDet < 540; ++iDet) {
    for (int iBin = 0; iBin < 25; ++iBin) {
      auto angleDiffSum = mHistogramEntriesSum[iDet * 25 + iBin];
      auto nEntries = mNEntriesPerBinSum[iDet * 25 + iBin];
      if (nEntries > 0) { // skip entries which have no entries; ?
        // add to the respective profile for fitting later on
        mFitFunctor.profiles[iDet]->Fill(2 * iBin - 25.f, angleDiffSum / nEntries, nEntries);
      }
    }
  }

  //----------------------------------------------------
  // Fitting
  //----------------------------------------------------
  printf("-------- Started fits\n");
  std::array<float, 540> laFitResults{};
  std::array<float, 540> vdFitResults{};
  for (int iDet = 0; iDet < 540; ++iDet) {
    mFitFunctor.currDet = iDet;
    ROOT::Fit::Fitter fitter;
    double paramsStart[2];
    paramsStart[0] = 0. * TMath::DegToRad();
    paramsStart[1] = 1.;
    fitter.SetFCN<o2::trd::FitFunctor>(2, mFitFunctor, paramsStart);
    fitter.Config().ParSettings(0).SetLimits(-0.7, 0.7);
    fitter.Config().ParSettings(0).SetStepSize(.01);
    fitter.Config().ParSettings(1).SetLimits(0., 3.);
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
    printf("Det %d: la=%f\tvd=%f\n", iDet, laFitResults[iDet] * TMath::RadToDeg(), vdFitResults[iDet]);
  }
  printf("-------- Finished fits\n");

  //----------------------------------------------------
  // Write
  //----------------------------------------------------
  std::unique_ptr<TFile> outFilePtr(TFile::Open("manualCalibFit.root", "RECREATE"));
  for (auto& p : mFitFunctor.profiles)
    p->Write();
}
