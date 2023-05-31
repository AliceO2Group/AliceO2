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

#include "MCHEvaluation/Draw.h"
#include <TCanvas.h>
#include <TF1.h>
#include <TFile.h>
#include <TGraphErrors.h>
#include <TH1.h>
#include <TLegend.h>
#include <TStyle.h>
#include <fmt/format.h>
#include <limits>
#include <memory>
#include <stdexcept>
#include <utility>

namespace
{
double CrystalBallSymmetric(double* xx, double* par)
{
  /// Crystal Ball definition

  /// par[0] = Normalization
  /// par[1] = mean
  /// par[2] = sigma
  /// par[3] = alpha = alpha'
  /// par[4] = n = n'

  double tp = fabs((xx[0] - par[1]) / par[2]);

  double absAlpha = fabs(par[3]);
  double ap = pow(par[4] / absAlpha, par[4]) * exp(-0.5 * absAlpha * absAlpha);
  double bp = par[4] / absAlpha - absAlpha;

  if (tp < absAlpha) {
    return par[0] * (exp(-0.5 * tp * tp)); // gaussian core
  } else {
    return par[0] * (ap / pow(bp + tp, par[4])); // left and right tails
  }
}

std::pair<double, double> GetSigma(TH1* h, int color)
{
  /// get the dispersion of the histo
  gStyle->SetOptFit(100);

  static TF1* fCrystalBall = new TF1("CrystalBall", CrystalBallSymmetric, -2., 2., 5);
  fCrystalBall->SetLineColor(color);

  if (h->GetEntries() < 10.) {
    return std::make_pair(0., 0.);
  }

  double sigmaTrk = 0.2;   // 2 mm
  double sigmaTrkCut = 4.; // 4 sigma

  // first fit
  double xMin = -0.5 * sigmaTrkCut * sigmaTrk;
  double xMax = 0.5 * sigmaTrkCut * sigmaTrk;
  fCrystalBall->SetRange(xMin, xMax);
  fCrystalBall->SetParameters(h->GetEntries(), 0., 0.1, 2., 1.5);
  fCrystalBall->SetParLimits(1, xMin, xMax);
  fCrystalBall->SetParLimits(2, 0., 1.);
  fCrystalBall->FixParameter(3, 1.e6);
  h->Fit(fCrystalBall, "RNQ");

  // rebin histo
  int rebin = 2;
  h->Rebin(rebin);

  // second fit
  fCrystalBall->SetParameter(0, fCrystalBall->GetParameter(0) * rebin);
  fCrystalBall->ReleaseParameter(3);
  fCrystalBall->SetParameter(3, 2.);
  fCrystalBall->SetParameter(4, 1.5);
  h->Fit(fCrystalBall, "RQ");

  return std::make_pair(fCrystalBall->GetParameter(2), fCrystalBall->GetParError(2));
}
} // namespace

namespace o2::mch::eval

{

auto padGridSize(const std::vector<TH1*>& histos)
{
  // find the optimal number of pads
  int nPadsx(1), nPadsy(1);
  while ((int)histos.size() > nPadsx * nPadsy) {
    if (nPadsx == nPadsy) {
      ++nPadsx;
    } else {
      ++nPadsy;
    }
  }
  return std::make_pair(nPadsx, nPadsy);
}

TCanvas* autoCanvas(const char* title, const char* name,
                    const std::vector<TH1*>& histos,
                    int* nPadsx, int* nPadsy)
{
  auto [nx, ny] = padGridSize(histos);
  if (nPadsx) {
    *nPadsx = nx;
  }
  if (nPadsy) {
    *nPadsy = ny;
  }
  TCanvas* c = new TCanvas(name, title, 10, 10, TMath::Max(nx * 300, 1200), TMath::Max(ny * 300, 900));
  c->Divide(nx, ny);
  return c;
}

void drawTrackResiduals(std::vector<TH1*>& histos, TCanvas* c)
{
  /// draw param2 - param1 histos
  gStyle->SetOptStat(111111);
  int nPadsx = (histos.size() + 2) / 3;
  if (!c) {
    c = new TCanvas("residuals", "residuals", 10, 10, nPadsx * 300, 900);
  }
  c->Divide(nPadsx, 3);
  int i(0);
  for (const auto& h : histos) {
    c->cd((i % 3) * nPadsx + i / 3 + 1);
    if (dynamic_cast<TH1F*>(h) == nullptr) {
      h->Draw("colz");
      gPad->SetLogz();
    } else {
      h->Draw();
      gPad->SetLogy();
    }
    ++i;
  }
}

void drawClusterResiduals(const std::array<std::vector<TH1*>, 5>& histos, TCanvas* c)
{
  drawClusterClusterResiduals(histos[4], "ClCl", c);
  drawClusterTrackResiduals(histos[0], histos[1], "AllTracks", c);
  drawClusterTrackResidualsSigma(histos[0], histos[1], "AllTracks", nullptr, c);
  drawClusterTrackResiduals(histos[2], histos[3], "SimilarTracks", c);
  drawClusterTrackResidualsSigma(histos[2], histos[3], "SimilarTracks", nullptr, c);
  drawClusterTrackResidualsRatio(histos[0], histos[1], "AllTracks", c);
  drawClusterTrackResidualsRatio(histos[2], histos[3], "SimilarTracks", c);
}

void drawClusterClusterResiduals(const std::vector<TH1*>& histos, const char* extension, TCanvas* c)
{
  /// draw cluster-cluster residuals
  gStyle->SetOptStat(1);
  int nPadsx = (histos.size() + 1) / 2;
  if (!c) {
    c = new TCanvas(Form("residual%s", extension), Form("residual%s", extension), 10, 10, nPadsx * 300, 600);
  }
  c->Divide(nPadsx, 2);
  int i(0);
  for (const auto& h : histos) {
    c->cd((i % 2) * nPadsx + i / 2 + 1);
    gPad->SetLogy();
    h->Draw();
    h->GetXaxis()->SetRangeUser(-0.02, 0.02);
    ++i;
  }
}

//_________________________________________________________________________________________________
void drawClusterTrackResiduals(const std::vector<TH1*>& histos1,
                               const std::vector<TH1*>& histos2, const char* extension,
                               TCanvas* c)
{
  /// draw cluster-track residuals
  gStyle->SetOptStat(1);
  int color[2] = {4, 2};
  int nPadsx = (histos1.size() + 1) / 2;
  if (!c) {
    c = new TCanvas(Form("residual%s", extension), Form("residual%s", extension), 10, 10, nPadsx * 300, 600);
  }
  c->Divide(nPadsx, 2);
  int i(0);
  for (const auto& h : histos1) {
    c->cd((i % 2) * nPadsx + i / 2 + 1);
    gPad->SetLogy();
    h->SetLineColor(color[0]);
    h->Draw();
    histos2[i]->SetLineColor(color[1]);
    histos2[i]->Draw("sames");
    ++i;
  }
  // add a legend
  TLegend* lHist = new TLegend(0.2, 0.65, 0.4, 0.8);
  lHist->SetFillStyle(0);
  lHist->SetBorderSize(0);
  lHist->AddEntry(histos1[0], "file 1", "l");
  lHist->AddEntry(histos2[0], "file 2", "l");
  c->cd(1);
  lHist->Draw("same");
}

void drawClusterTrackResidualsSigma(const std::vector<TH1*>& histos1,
                                    const std::vector<TH1*>& histos2, const char* extension,
                                    TCanvas* c1, TCanvas* c2)
{
  /// draw cluster-track residuals and fit them to extract the resolution
  gStyle->SetOptStat(1);

  TGraphErrors* g[2][2] = {nullptr};
  const char* dir[2] = {"X", "Y"};
  int color[2] = {4, 2};
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 2; ++j) {
      g[i][j] = new TGraphErrors(6);
      g[i][j]->SetName(Form("sigma%s%s%d", dir[i], extension, j));
      g[i][j]->SetTitle(Form("#sigma%s per station;station ID;#sigma%s (cm)", dir[i], dir[i]));
      g[i][j]->SetMarkerStyle(kFullDotLarge);
      g[i][j]->SetMarkerColor(color[j]);
      g[i][j]->SetLineColor(color[j]);
    }
  }

  int nPadsx = (histos1.size() + 1) / 2;
  if (!c1) {
    c1 = new TCanvas(Form("residual%sFit", extension), Form("residual%sFit", extension), 10, 10, nPadsx * 300, 600);
  }
  c1->Divide(nPadsx, 2);
  int i(0);
  double ymax[2] = {0., 0.};
  double ymin[2] = {std::numeric_limits<double>::max(), std::numeric_limits<double>::max()};
  for (auto h : histos1) {
    c1->cd((i % 2) * nPadsx + i / 2 + 1);
    gPad->SetLogy();
    auto h1 = static_cast<TH1*>(h->Clone());
    h1->SetLineColor(color[0]);
    h1->Draw();
    auto res1 = GetSigma(h1, color[0]);
    g[i % 2][0]->SetPoint(i / 2, i * 1. / 2 + 1., res1.first);
    g[i % 2][0]->SetPointError(i / 2, 0., res1.second);
    ymin[i % 2] = std::min(ymin[i % 2], res1.first - res1.second - 0.001);
    ymax[i % 2] = std::max(ymax[i % 2], res1.first + res1.second + 0.001);
    auto h2 = static_cast<TH1*>(histos2[i]->Clone());
    h2->SetLineColor(color[1]);
    h2->Draw("sames");
    auto res2 = GetSigma(h2, color[1]);
    g[i % 2][1]->SetPoint(i / 2, i * 1. / 2 + 1., res2.first);
    g[i % 2][1]->SetPointError(i / 2, 0., res2.second);
    ymin[i % 2] = std::min(ymin[i % 2], res2.first - res2.second - 0.001);
    ymax[i % 2] = std::max(ymax[i % 2], res2.first + res2.second + 0.001);
    printf("%s: %f ± %f cm --> %f ± %f cm\n", h->GetName(), res1.first, res1.second, res2.first, res2.second);
    ++i;
  }

  if (!c2) {
    c2 = new TCanvas(Form("sigma%s", extension), Form("sigma%s", extension), 10, 10, 600, 300);
  }
  c2->Divide(2, 1);
  c2->cd(1);
  g[0][0]->Draw("ap");
  g[0][0]->GetYaxis()->SetRangeUser(ymin[0], ymax[0]);
  g[0][1]->Draw("p");
  c2->cd(2);
  g[1][0]->Draw("ap");
  g[1][0]->GetYaxis()->SetRangeUser(ymin[1], ymax[1]);
  g[1][1]->Draw("p");

  // add a legend
  TLegend* lHist = new TLegend(0.2, 0.65, 0.4, 0.8);
  lHist->SetFillStyle(0);
  lHist->SetBorderSize(0);
  lHist->AddEntry(histos1[0], "file 1", "l");
  lHist->AddEntry(histos2[0], "file 2", "l");
  c1->cd(1);
  lHist->Draw("same");
}

void drawPlainHistosAtVertex(const std::array<std::vector<TH1*>, 2>& histos, TCanvas* c)
{
  if (!c) {
    c = autoCanvas("histos", "histos", histos[0]);
  }

  for (int i = 0; i < (int)histos[0].size(); ++i) {
    c->cd(i + 1);
    gPad->SetLogy();
    histos[0][i]->SetStats(false);
    histos[0][i]->SetLineColor(4);
    histos[0][i]->Draw();
    histos[1][i]->SetLineColor(2);
    histos[1][i]->Draw("same");
  }

  // add a legend
  TLegend* lHist = new TLegend(0.5, 0.65, 0.9, 0.8);
  lHist->SetFillStyle(0);
  lHist->SetBorderSize(0);
  lHist->AddEntry(histos[0][0], Form("%g tracks in file 1", histos[0][0]->GetEntries()), "l");
  lHist->AddEntry(histos[1][0], Form("%g tracks in file 2", histos[1][0]->GetEntries()), "l");
  c->cd(1);
  lHist->Draw("same");
}

void drawDiffHistosAtVertex(const std::array<std::vector<TH1*>, 2>& histos, TCanvas* c)
{
  if (!c) {
    c = autoCanvas("differences", "histos2 - histos1", histos[0]);
  }

  // draw differences
  for (int i = 0; i < (int)histos[0].size(); ++i) {
    c->cd(i + 1);
    TH1F* hDiff = static_cast<TH1F*>(histos[1][i]->Clone());
    hDiff->Add(histos[0][i], -1.);
    hDiff->SetStats(false);
    hDiff->SetLineColor(2);
    hDiff->Draw();
  }
}

void drawClusterTrackResidualsRatio(const std::vector<TH1*>& histos1,
                                    const std::vector<TH1*>& histos2,
                                    const char* extension,
                                    TCanvas* c)
{
  /// draw ratios of cluster-track residuals

  int nPadsx = (histos1.size() + 1) / 2;
  if (!c) {
    c = new TCanvas(Form("ratio%s", extension), Form("ratio%s", extension), 10, 10, nPadsx * 300, 600);
  }
  c->Divide(nPadsx, 2);
  int i(0);
  for (const auto& h : histos2) {
    c->cd((i % 2) * nPadsx + i / 2 + 1);
    TH1* hRat = new TH1F(*static_cast<TH1F*>(h));
    hRat->Rebin(4);
    auto h1 = static_cast<TH1F*>(histos1[i]->Clone());
    h1->Rebin(4);
    hRat->Divide(h1);
    delete h1;
    hRat->SetStats(false);
    hRat->SetLineColor(2);
    hRat->Draw();
    hRat->GetXaxis()->SetRangeUser(-0.5, 0.5);
    ++i;
  }
}

void drawRatioHistosAtVertex(const std::array<std::vector<TH1*>, 2>& histos, TCanvas* c)
{
  //  draw ratios
  if (!c) {
    c = autoCanvas("ratios", "histos2 / histos1", histos[0]);
  }

  for (int i = 0; i < (int)histos[0].size(); ++i) {
    c->cd(i + 1);
    TH1F* hRat = static_cast<TH1F*>(histos[1][i]->Clone());
    hRat->Divide(histos[0][i]);
    hRat->SetStats(false);
    hRat->SetLineColor(2);
    hRat->Draw();
  }
}

void drawHistosAtVertex(const std::array<std::vector<TH1*>, 2>& histos,
                        TCanvas* c)
{
  /// Draw histograms at vertex and differences between the 2 inputs

  drawPlainHistosAtVertex(histos, c);
  drawDiffHistosAtVertex(histos, c);
  drawRatioHistosAtVertex(histos, c);
}

void drawComparisonsAtVertex(const std::array<std::vector<TH1*>, 5> histos, TCanvas* c)
{
  /// draw comparison histograms at vertex

  if (!c) {
    c = autoCanvas("comparisons", "comparisons", histos[0]);
  }
  for (int i = 0; i < (int)histos[0].size(); ++i) {
    c->cd(i + 1);
    gPad->SetLogy();
    histos[0][i]->SetStats(false);
    histos[0][i]->SetLineColor(1);
    histos[0][i]->SetMinimum(0.5);
    histos[0][i]->Draw();
    histos[1][i]->SetLineColor(4);
    histos[1][i]->Draw("same");
    histos[2][i]->SetLineColor(877);
    histos[2][i]->Draw("same");
    histos[3][i]->SetLineColor(3);
    histos[3][i]->Draw("same");
    histos[4][i]->SetLineColor(2);
    histos[4][i]->Draw("same");
  }

  // add a legend
  TLegend* lHist = new TLegend(0.5, 0.5, 0.9, 0.8);
  lHist->SetFillStyle(0);
  lHist->SetBorderSize(0);
  lHist->AddEntry(histos[0][0], Form("%g tracks identical", histos[0][0]->GetEntries()), "l");
  lHist->AddEntry(histos[1][0], Form("%g tracks similar (1)", histos[1][0]->GetEntries()), "l");
  lHist->AddEntry(histos[2][0], Form("%g tracks similar (2)", histos[2][0]->GetEntries()), "l");
  lHist->AddEntry(histos[3][0], Form("%g tracks additional", histos[3][0]->GetEntries()), "l");
  lHist->AddEntry(histos[4][0], Form("%g tracks missing", histos[4][0]->GetEntries()), "l");
  c->cd(1);
  lHist->Draw("same");
}

void drawAll(const char* filename)
{
  TFile* f = TFile::Open(filename);

  f->ls();

  std::vector<TH1*>* trackResidualsAtFirstCluster = reinterpret_cast<std::vector<TH1*>*>(f->GetObjectUnchecked("trackResidualsAtFirstCluster"));
  if (trackResidualsAtFirstCluster) {
    drawTrackResiduals(*trackResidualsAtFirstCluster);
  } else {
    std::cerr << "could not read trackResidualsAtFirstCluster vector of histogram from file\n";
  }

  std::array<std::vector<TH1*>, 5>* clusterResiduals = reinterpret_cast<std::array<std::vector<TH1*>, 5>*>(f->GetObjectUnchecked("clusterResiduals"));
  drawClusterResiduals(*clusterResiduals);

  std::array<std::vector<TH1*>, 2>* histosAtVertex = reinterpret_cast<std::array<std::vector<TH1*>, 2>*>(f->GetObjectUnchecked("histosAtVertex"));
  drawHistosAtVertex(*histosAtVertex);

  std::array<std::vector<TH1*>, 5>* comparisonsAtVertex = reinterpret_cast<std::array<std::vector<TH1*>, 5>*>(f->GetObjectUnchecked("comparisonsAtVertex"));
  drawComparisonsAtVertex(*comparisonsAtVertex);
}
} // namespace o2::mch::eval
