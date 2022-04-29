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

#define BOOST_TEST_MODULE Test NDRegression2DGaus
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
using o2::nd_regression::NDRegression;
using std::make_shared;
using std::make_unique;

void PlotData(TH1F* hData, TString xTitle = "xTitle", TString yTitle = "yTitle", Color_t color = kBlack, TString zTitle = "zTitle", Double_t rms = 999999., Double_t eRms = 0., Double_t mean = 999999., Double_t eMean = 0.)
{
  //
  //
  // gStyle->SetPadRightMargin(0.05);
  // gStyle->SetPadTopMargin(0.05);
  // gStyle->SetPadLeftMargin(0.24);
  // gStyle->SetPadBottomMargin(0.22);
  // gStyle->SetPadTickX(1);
  // gStyle->SetPadTickY(1);
  // gStyle->SetPadGridX(1);
  // gStyle->SetPadGridY(1);
  // gStyle->SetOptStat(0);
  gPad->SetRightMargin(0.05);
  gPad->SetTopMargin(0.05);
  gPad->SetLeftMargin(0.14);
  gPad->SetBottomMargin(0.12);
  gPad->SetTicks(1);
  gPad->SetGrid(1);
  gPad->SetObjectStat(0);
  //
  if (color == (kRed + 2)) {
    hData->SetMarkerStyle(20);
  }
  if (color == (kBlue + 2)) {
    hData->SetMarkerStyle(21);
  }
  if (color == (kGreen + 2)) {
    hData->SetMarkerStyle(22);
  }
  hData->SetMarkerSize(0.7);

  hData->SetMarkerColor(color);
  hData->SetLineColor(color);
  hData->GetXaxis()->SetTitle(xTitle.Data());
  hData->GetYaxis()->SetTitle(yTitle.Data());
  hData->GetZaxis()->SetTitle(zTitle.Data());
  hData->GetXaxis()->SetTitleOffset(1.2);
  hData->GetXaxis()->SetTitleSize(0.05);
  hData->GetYaxis()->SetTitleOffset(1.3);
  hData->GetYaxis()->SetTitleSize(0.05);
  hData->GetXaxis()->SetLabelSize(0.035);
  hData->GetYaxis()->SetLabelSize(0.035);
  hData->GetXaxis()->SetDecimals();
  hData->GetYaxis()->SetDecimals();
  hData->GetZaxis()->SetDecimals();
  hData->Sumw2();
  hData->Draw("pe1 same");
  // hData->Draw("l same");

  if (mean != 999999.) {
    TPaveText* text1 = new TPaveText(0.21, 0.82, 0.51, 0.92, "NDC");
    text1->SetTextFont(43);
    text1->SetTextSize(30.);
    text1->SetBorderSize(1);
    text1->SetFillColor(kWhite);
    text1->AddText(Form("Mean: %0.2f #pm %0.2f", mean, eMean));
    text1->AddText(Form("RMS: %0.2f #pm %0.2f", rms, eRms));
    text1->Draw();
  }
  if (rms != 999999. && mean == 999999.) {
    TPaveText* text1 = new TPaveText(0.21, 0.87, 0.51, 0.92, "NDC");
    text1->SetTextFont(43);
    text1->SetTextSize(30.);
    text1->SetBorderSize(1);
    text1->SetFillColor(kWhite);
    text1->AddText(Form("RMS: %0.2f", rms));
    text1->Draw();
  }
}

BOOST_AUTO_TEST_CASE(NDRegression2DGaus_test)
{
  auto pfitNDGaus0 = make_unique<NDRegression>("pfitNDGaus0", "pfitNDGaus0");

  auto success = pfitNDGaus0->init();
  BOOST_CHECK(success == true);

  //
  // 0.) Initialization of variables and THn
  //
  Int_t ndim = 2;
  Double_t err = 0.1;

  o2::utils::TreeStreamRedirector pcstreamIn("fitNDLocalTestInput.root", "recreate");
  auto pcstreamOutGaus0 = make_shared<o2::utils::TreeStreamRedirector>("fitNDLocalTestOutputGaus0.root", "recreate");

  Double_t* xyz = new Double_t[ndim];
  Double_t* sxyz = new Double_t[ndim];
  Int_t* nbins = new Int_t[ndim];
  Double_t* xmin = new Double_t[ndim];
  Double_t* xmax = new Double_t[ndim];
  TString** chxyz = new TString*[ndim];
  for (Int_t idim = 0; idim < ndim; idim++) {
    chxyz[idim] = new TString(TString::Format("xyz%d=", idim).Data());
    nbins[idim] = 70;
    xmin[idim] = -1;
    xmax[idim] = 2;
  }

  auto pformula = make_shared<TFormula>("pformula", "cos(7*x[0])*cos(3*x[1])+pow(x[0], x[1])");
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
  std::cout << "testfile done\n";

  TFile inpf("fitNDLocalTestInput.root");
  BOOST_CHECK(!inpf.IsZombie());
  auto treeIn = (TTree*)(inpf.GetFile()->Get("testInput"));
  BOOST_CHECK(treeIn);

  pfitNDGaus0->SetStreamer(pcstreamOutGaus0);

  
  success = pfitNDGaus0->SetHistogram((THn*)((hN->Clone())));
  BOOST_CHECK(success);

  success = pfitNDGaus0->MakeFit(treeIn, "val+noise:err", "xyz0:xyz1","Entry$%2==1", "0.02:0.02","2:2",0.0001);  // sample Gaussian1
  BOOST_CHECK(success);

  std::cout << "Now drawing...\n";

  pfitNDGaus0->AddVisualCorrection(pfitNDGaus0.get(),1);

  TObjArray * array = NDRegression::GetVisualCorrections();
  for (Int_t i=0; i<array->GetEntries(); i++){
    NDRegression * regression = ( NDRegression *)array->At(i);
    if (regression==NULL) continue;
    regression->AddVisualCorrection(regression);
    Int_t hashIndex = regression->GetVisualCorrectionIndex();
    treeIn->SetAlias( regression->GetName(), TString::Format("o2::nd_regression::NDRegression::GetCorrND(%d,xyz0,xyz1+0)",hashIndex).Data());
  }
  pcstreamOutGaus0.reset();
  std::cout << "Entries: " << array->GetEntries() << std::endl;

  
  TCanvas * canvasGaus = new TCanvas("canvasGaus","canvasGaus",800,800);
  treeIn->Draw("val>>inputData(71,-1.1,2.1)","","");
  TH1F   *inputData = (TH1F*)gPad->GetPrimitive("inputData");
  treeIn->Draw("(o2::nd_regression::NDRegression::GetCorrND(1,xyz0,xyz1))>>gaus(71,-1.1,2.1)","","");
  // treeIn->Draw("(o2::nd_regression::NDRegression::GetCorrND(3,xyz0,xyz1)-o2::nd_regression::NDRegression::GetCorrND(2,xyz0,xyz1))/sqrt(o2::nd_regression::NDRegression::GetCorrNDError(3,xyz0,xyz1)**2+o2::nd_regression::NDRegression::GetCorrNDError(2,xyz0,xyz1)**2)>>ideal(401,-20.5,20.5)","","");
  TH1F   *gaus = (TH1F*)gPad->GetPrimitive("gaus");
  Double_t meanGaus = treeIn->GetHistogram()->GetMean();
  Double_t meanGausErr = treeIn->GetHistogram()->GetMeanError();
  Double_t rmsGaus = treeIn->GetHistogram()->GetRMS();
  Double_t rmsGausErr = treeIn->GetHistogram()->GetRMSError();
  if (TMath::Abs(meanGaus) <10*meanGausErr) {
    ::Info( "NDRegression2DGausTest","mean pull OK %3.3f\t+-%3.3f", meanGaus, meanGausErr);
  }else{
    ::Error( "NDRegressionTest","mean pull NOT OK %3.3f\t+-%3.3f", meanGaus, meanGausErr);
  }
  if (rmsGaus<1+rmsGausErr) {
    ::Info( "NDRegressionTest"," rms pull OK %3.3f\t+-%3.3f", rmsGaus, rmsGausErr);
  }else{
    ::Error( "NDRegressionTest"," rms pull NOT OK %3.3f\t+-%3.3f", rmsGaus, rmsGausErr);
  }
  
  
  
  inputData->Draw("same");
  PlotData(gaus,"Gaus","counts (arb. units)",kRed+2,"zTitle",rmsGaus,rmsGausErr,meanGaus,meanGausErr);
  canvasGaus->SaveAs("NDRegressionTest.canvasGausTest.png");

  inpf.Close();
}
