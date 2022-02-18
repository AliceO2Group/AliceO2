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

#define BOOST_TEST_MODULE Test NDRegression2DIdeal
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>
#include "NDRegression/NDRegression.h"
#include "TRandom.h"
#include "TFile.h"
#include "THn.h"

// namespace o2::utils
// {
// class TreeStreamRedirector;
// }

// using o2::utils::TreeStreamRedirector;
using std::make_shared;
using std::make_unique;

BOOST_AUTO_TEST_CASE(NDRegression2DIdeal_test)
{
  auto pfitNDIdeal = make_unique<o2::nd_regression::NDRegression>("pfitNDIdeal", "pfitNDIdeal");
  auto success = pfitNDIdeal->init();

  BOOST_CHECK(success == true);

  //
  // 0.) Initialization of variables and THn
  //
  Int_t ndim = 2;
  Double_t err = 0.1;

  o2::utils::TreeStreamRedirector pcstreamIn("fitNDLocalTestInput.root", "recreate");
  auto pcstreamOutIdeal = make_shared<o2::utils::TreeStreamRedirector>("fitNDLocalTestOutputIdeal.root", "recreate");

  Double_t* xyz = new Double_t[ndim];
  Double_t* sxyz = new Double_t[ndim];
  Int_t* nbins = new Int_t[ndim];
  Double_t* xmin = new Double_t[ndim];
  Double_t* xmax = new Double_t[ndim];
  TString** chxyz = new TString*[ndim];
  for (Int_t idim = 0; idim < ndim; idim++) {
    chxyz[idim] = new TString(TString::Format("xyz%d=", idim).Data());
    nbins[idim] = 40;
    xmin[idim] = 0;
    xmax[idim] = 1;
  }

  auto pformula = make_shared<TFormula>("pformula", "cos(7*x[0]/pi)*sin(11*x[1]/pi)");
  auto hN = make_shared<THnF>("exampleFit", "exampleFit", ndim, nbins, xmin, xmax);
  //
  // 1.) generate random input points
  //
  for (Int_t ipoint = 0; ipoint < 1e4; ipoint++) {
    for (Int_t idim = 0; idim < ndim; idim++) {
      xyz[idim] = gRandom->Rndm();
    }
    Double_t value = pformula->EvalPar(xyz, 0);
    Double_t noise = gRandom->Gaus() * err;
    Double_t noise2 = noise * (1 + (gRandom->Rndm() < 0.1) * 100); // noise with 10 percent of outliers
    Double_t noiseBreit = gRandom->BreitWigner() * err;
    pcstreamIn << "testInput"
               << "val=" << value << "err=" << err << "noise=" << noise << // gausian noise
      "noise2=" << noise2 <<                                               // gausian noise + 10% of outliers
      "noiseBreit=" << noiseBreit;
    for (Int_t idim = 0; idim < ndim; idim++) {
      pcstreamIn << "testInput" << chxyz[idim]->Data() << xyz[idim];
    }
    pcstreamIn << "testInput"
               << "\n";
  }
  pcstreamIn.Close();
  std::cout << "testfile done";

  TFile inpf("fitNDLocalTestInput.root");
  BOOST_CHECK(!inpf.IsZombie());
  auto treeIn = (TTree*)(inpf.GetFile()->Get("testInput"));
  BOOST_CHECK(treeIn);

  pfitNDIdeal->SetStreamer(pcstreamOutIdeal);

  success = pfitNDIdeal->SetHistogram((THn*)((hN->Clone())));
  BOOST_CHECK(success);

  success = pfitNDIdeal->MakeFit(treeIn, "val:err", "xyz0:xyz1", "Entry$%2==1", "0.02:0.02", "2:2", 0.0001);
  BOOST_CHECK(success);

  treeIn->Draw("(AliNDLocalRegression::GetCorrND(5,xyz0,xyz1)-AliNDLocalRegression::GetCorrND(4,xyz0,xyz1))/sqrt(AliNDLocalRegression::GetCorrNDError(4,xyz0,xyz1)**2+AliNDLocalRegression::GetCorrNDError(5,xyz0,xyz1)**2)>>pullsBreiWigner54(401,-20.5,20.5)", "", "");
  TH1F* pullsBreiWigner54 = (TH1F*)gPad->GetPrimitive("pullsBreiWigner54");
  Double_t meanPullsBreitWigner = treeIn->GetHistogram()->GetMean();
  Double_t meanPullsBreitWignerErr = treeIn->GetHistogram()->GetMeanError();
  Double_t rmsPullsBreitWigner = treeIn->GetHistogram()->GetRMS();
  Double_t rmsPullsBreitWignerErr = treeIn->GetHistogram()->GetRMSError();
  if (TMath::Abs(meanPullsBreitWigner) < 10 * meanPullsBreitWignerErr) {
    ::Info("AliNDLocalRegressionTest::UnitTestBreitWignerNoise", "mean pull OK %3.3f\t+-%3.3f", meanPullsBreitWigner, meanPullsBreitWignerErr);
  } else {
    ::Error("AliNDLocalRegressionTest::UnitTestBreitWignerNoise", "mean pull NOT OK %3.3f\t+-%3.3f", meanPullsBreitWigner, meanPullsBreitWignerErr);
  }
  if (rmsPullsBreitWigner < 1 + rmsPullsBreitWignerErr) {
    ::Info("AliNDLocalRegressionTest::UnitTestBreitWignerNoise", " rms pull OK %3.3f\t+-%3.3f", rmsPullsBreitWigner, rmsPullsBreitWignerErr);
  } else {
    ::Error("AliNDLocalRegressionTest::UnitTestBreitWignerNoise", " rms pull NOT OK %3.3f\t+-%3.3f", rmsPullsBreitWigner, rmsPullsBreitWignerErr);
  }
  PlotData(pullsBreiWigner54, "BreitWigner pulls", "counts (arb. units)", kBlue + 2, "zTitle", rmsPullsBreitWigner, rmsPullsBreitWignerErr, meanPullsBreitWigner, meanPullsBreitWignerErr);
  canvasUnitBreitWigner->SaveAs("AliNDLocalRegressionTest.canvasUnitBreitWigner.png");

  inpf.Close();
}
