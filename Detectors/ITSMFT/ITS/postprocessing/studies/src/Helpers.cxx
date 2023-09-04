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

#include <cassert>

// o2 includes
#include "ITSStudies/Helpers.h"
#include "TLegend.h"

using namespace o2::its::study;

//______________________________________________________________________________
std::vector<double> helpers::makeLogBinning(const int nbins, const double min, const double max)
{
  assert(min > 0);
  assert(min < max);

  std::vector<double> binLim(nbins + 1);

  const double expMax = std::log(max / min);
  const double binWidth = expMax / nbins;

  binLim[0] = min;
  binLim[nbins] = max;

  for (Int_t i = 1; i < nbins; ++i) {
    binLim[i] = min * std::exp(i * binWidth);
  }

  return binLim;
}

//______________________________________________________________________________
void helpers::setStyleHistogram1D(TH1F& histo, int color)
{
  // common and 1D case
  histo.SetStats(0);
  histo.SetMinimum(20);
  histo.SetMaximum(300);
  histo.SetLineColor(color);
  histo.SetLineWidth(1);
  histo.SetMarkerColor(color);
  histo.SetMarkerSize(0.05);
  histo.SetMarkerStyle(1);
  histo.SetTitle("");
}

//______________________________________________________________________________
void helpers::setStyleHistogram1D(TH1F& histo, int color, TString title, TString titleYaxis, TString titleXaxis)
{
  // common and 1D case
  helpers::setStyleHistogram1D(histo, color);
  histo.GetXaxis()->SetTitle(titleXaxis.Data());
  histo.GetYaxis()->SetTitle(titleYaxis.Data());
  histo.GetXaxis()->SetTitleOffset(1.4);
  histo.GetYaxis()->SetTitleOffset(1.2);
}

//______________________________________________________________________________
void helpers::setStyleHistogram1DMeanValues(TH1F& histo, int color)
{
  // common and 1D case
  helpers::setStyleHistogram1D(histo, color);
  histo.SetMinimum(-15.);
  histo.SetMaximum(15.);
  histo.GetXaxis()->SetTitleOffset(1.4);
  histo.GetYaxis()->SetTitleOffset(1.2);
}

//______________________________________________________________________________
void helpers::setStyleHistogram2D(TH2F& histo)
{
  // common and 1D case
  histo.SetStats(0);
  histo.GetYaxis()->SetRangeUser(-600, 600);
  histo.GetXaxis()->SetTitleOffset(1.2);
  histo.GetYaxis()->SetTitleOffset(1.2);
}

//______________________________________________________________________________
TCanvas* helpers::prepareSimpleCanvas2Histograms(TH1F& h1, int color1, TH1F& h2, int color2)
{
  TCanvas* c1 = new TCanvas();
  c1->SetLogy();
  c1->SetGridy();
  c1->SetLogx();
  c1->SetGridx();
  setStyleHistogram1D(h1, color1);
  helpers::setStyleHistogram1D(h2, color2);
  h1.Draw();
  h2.Draw("same");
  return c1;
}

//______________________________________________________________________________
TCanvas* helpers::prepareSimpleCanvas2Histograms(TH1F& h1, int color1, TString nameHisto1, TH1F& h2, int color2, TString nameHisto2, bool logScale)
{
  TCanvas* c1 = new TCanvas();
  if (logScale) {
    c1->SetLogy(); // c1->SetGridy();
    c1->SetLogx(); // c1->SetGridx();
  }
  TString direction = "";
  TString histoName1 = h1.GetName();
  TString histoName2 = h2.GetName();
  if (histoName1.Contains("Xy")) {
    direction = "XY";
  }
  if (histoName1.Contains("Z")) {
    direction = "Z";
  }
  if ((histoName1.Contains("Xy")) && (histoName2.Contains("Z"))) {
    direction = "";
  }
  helpers::setStyleHistogram1D(h1, color1, "", Form("Pointing Resolution %s (#mum)", direction.Data()), h1.GetXaxis()->GetName());
  helpers::setStyleHistogram1D(h2, color2, "", Form("Pointing Resolution %s (#mum)", direction.Data()), h2.GetXaxis()->GetName());
  TLegend* leg = new TLegend(0.6, 0.3, 0.8, 0.5);
  leg->SetBorderSize(0);
  leg->SetFillStyle(0);
  leg->AddEntry(&h1, nameHisto1.Data(), "lp");
  leg->AddEntry(&h2, nameHisto2.Data(), "lp");
  h1.Draw();
  h2.Draw("same");
  leg->Draw("same");
  return c1;
}

//______________________________________________________________________________
TCanvas* helpers::prepareSimpleCanvas2Histograms(TH1F& h1, int color1, TString nameHisto1, TH1F& h2, int color2, TString nameHisto2, TString intRate)
{
  TCanvas* c1 = new TCanvas();
  c1->SetLogy(); // c1->SetGridy();
  c1->SetLogx(); // c1->SetGridx();
  helpers::setStyleHistogram1D(h1, color1);
  helpers::setStyleHistogram1D(h2, color2);
  TLegend* leg = new TLegend(0.2, 0.3, 0.5, 0.5);
  leg->SetBorderSize(0);
  leg->SetFillStyle(0);
  leg->AddEntry(&h1, nameHisto1.Data(), "lp");
  leg->AddEntry(&h2, nameHisto2.Data(), "lp");
  h1.Draw();
  h2.Draw("same");
  leg->Draw("same");
  TPaveText* paveText;
  if (!intRate) {
    helpers::paveTextITS(paveText, intRate);
  }
  return c1;
}

//______________________________________________________________________________
TCanvas* helpers::prepareSimpleCanvas2DcaMeanValues(TH1F& h1, int color1, TString nameHisto1, TH1F& h2, int color2, TString nameHisto2)
{
  TCanvas* c1 = new TCanvas();
  c1->SetLogx();
  helpers::setStyleHistogram1DMeanValues(h1, color1);
  helpers::setStyleHistogram1DMeanValues(h2, color2);
  TLegend* leg = new TLegend(0.2, 0.15, 0.5, 0.35);
  leg->SetBorderSize(0);
  leg->SetFillStyle(0);
  leg->AddEntry(&h1, nameHisto1.Data(), "lp");
  leg->AddEntry(&h2, nameHisto2.Data(), "lp");
  h1.Draw();
  h2.Draw("same");
  leg->Draw("same");
  return c1;
}

//______________________________________________________________________________
TCanvas* helpers::plot2DwithMeanAndSigma(TH2F& h2D, TH1F& hMean, TH1F& hSigma, int color)
{
  TCanvas* c1 = new TCanvas();
  helpers::setStyleHistogram2D(h2D);
  helpers::setStyleHistogram1D(hSigma, color);
  h2D.Draw("colz");
  hMean.Draw("same");
  TGraphAsymmErrors* gSigma;
  ConvertTH1ToTGraphAsymmError(hMean, hSigma, gSigma);
  gSigma->SetLineColor(kRed);
  gSigma->SetFillStyle(0);
  gSigma->Draw("E2same");
  return c1;
}

//______________________________________________________________________________
void helpers::paveTextITS(TPaveText* pave, TString intRate)
{
  pave->SetFillStyle(0);
  pave->SetBorderSize(0);
  pave->SetFillColor(0);
  pave->SetTextFont(53);
  pave->SetTextSize(12);
  pave->AddText("ALICE");
  pave->AddText("Run3 ITS Performaces");
  pave->AddText(Form("Interaction Rate = %s", intRate.Data()));
}

//______________________________________________________________________________
void helpers::ConvertTH1ToTGraphAsymmError(TH1F& hMean, TH1F& hSigma, TGraphAsymmErrors*& gr)
{
  const Int_t nbinsxx = hMean.GetNbinsX() + 1;
  Double_t x[nbinsxx], y[nbinsxx], ex1[nbinsxx], ex2[nbinsxx], ey1[nbinsxx], ey2[nbinsxx];

  for (int i = 0; i < nbinsxx; i++) {
    x[i] = hMean.GetBinCenter(i);
    y[i] = hMean.GetBinContent(i);
    ex1[i] = hMean.GetBinCenter(i) - hMean.GetBinLowEdge(i);
    ex2[i] = ex1[i];
    ey1[i] = hSigma.GetBinContent(i);
    ey2[i] = hSigma.GetBinContent(i);
  }

  gr = new TGraphAsymmErrors(nbinsxx, x, y, ex1, ex2, ey1, ey2);
  return;
}
