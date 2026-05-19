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

#if !defined(__CLING__) || defined(__ROOTCLING__)
#include <TFile.h>
#include <TF1.h>
#include <TH1F.h>
#include <TH2F.h>
#include <TCanvas.h>
#include <TTree.h>
#endif

void PlotResiduals(const char* fname = "its3TrackStudy.root")
{
  auto f = TFile::Open(fname);
  auto res = f->Get<TTree>("res");

  const int nLay = 8;
  const int nVar = 6;
  const char* vars[nVar] = {"dYInt", "dZInt", "dYIn", "dZIn", "dYOut", "dZOut"};
  const char* titles[nVar] = {"d_{Y} (#mum) (weighted)", "d_{Z} (#mum) (weighted)", "d_{Y} (#mu) (inward)", "d_{Y} (#mu) (inward)", "d_{Y} (#mu) (outward)", "d_{Z} (#mu) (outward)"};

  TCanvas* canvs[nVar];
  for (int iv = 0; iv < nVar; iv++) {
    canvs[iv] = new TCanvas(vars[iv], Form("%s residuals", vars[iv]), 800, 1600);
    canvs[iv]->Divide(2, 4);
  }

  for (int iv = 0; iv < nVar; iv++) {
    canvs[iv]->cd(0);
    for (int lay = -1; lay <= 6; lay++) {
      canvs[iv]->cd(lay + 2);
      gPad->SetRightMargin(0.13);

      TString hname = Form("h_%s_lay%d", vars[iv], lay + 1);
      TString expr = Form("%s*10000:phi>>%s(100,0,6.283,100,-100,100)", vars[iv], hname.Data());
      TString sel = Form("lay==%d", lay);
      res->Draw(expr, sel, "col");

      auto* h = (TH2F*)gDirectory->Get(hname);
      h->SetTitle(Form("Layer %d ;#phi (rad);%s", lay, titles[iv]));
      h->GetZaxis()->SetLabelSize(0.035);

      // fit y-slices with gaussian
      h->FitSlicesY(nullptr, 0, -1, 5);
      auto* hMean = (TH1D*)gDirectory->Get(Form("%s_1", hname.Data()));
      auto* hSigma = (TH1D*)gDirectory->Get(Form("%s_2", hname.Data()));

      for (auto* hh : {hMean, hSigma}) {
        hh->SetMarkerStyle(20);
        hh->SetMarkerSize(0.6);
      }
      hMean->SetMarkerColor(kRed);
      hMean->SetLineColor(kRed);
      hSigma->SetMarkerColor(kOrange + 1);
      hSigma->SetLineColor(kOrange + 1);
      hMean->Draw("same");
      hSigma->Draw("same");
    }
    canvs[iv]->SaveAs(Form("%s.png", canvs[iv]->GetName()));
  }
}
