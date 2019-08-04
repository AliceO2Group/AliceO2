// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#if !defined(__CLING__) || defined(__ROOTCLING__)
#include "TROOT.h"
#include "TFile.h"
#include "TPCBase/CalDet.h"
#include "TH1F.h"
#include "TH2F.h"
#include "TCanvas.h"
#include "TPCBase/Painter.h"
#endif

std::tuple<TH1*, TH1*, TH1*, TH1*> getNoiseAndPedestalHistogram(const TString pedestalFile, int roc)
{
  using namespace o2::tpc;
  TFile f(pedestalFile);
  gROOT->cd();

  // ===| load noise and pedestal from file |===
  CalDet<float> dummy;
  CalDet<float>* pedestal = nullptr;
  CalDet<float>* noise = nullptr;
  f.GetObject("Pedestals", pedestal);
  f.GetObject("Noise", noise);
  const auto& rocPedestal = pedestal->getCalArray(roc);
  const auto& rocNoise = noise->getCalArray(roc);

  // ===| histograms for noise and pedestal |===
  auto hPedestal = new TH1F("hPedestal", Form("Pedestal distribution (%s);ADC value", pedestalFile.Data()), 100, 50, 150);
  hPedestal->SetDirectory(nullptr);

  auto hNoise = new TH1F("hNoise", Form("Noise distribution (%s);ADC value", pedestalFile.Data()), 100, 0, 5);
  hNoise->SetDirectory(nullptr);

  auto hPedestal2D = painter::getHistogram2D(rocPedestal);
  hPedestal2D->SetStats(0);
  hPedestal2D->SetDirectory(nullptr);
  hPedestal2D->SetTitle(Form("%s (%s)", hPedestal2D->GetTitle(), pedestalFile.Data()));

  auto hNoise2D = painter::getHistogram2D(rocNoise);
  hNoise2D->SetStats(0);
  hNoise2D->SetDirectory(nullptr);
  hNoise2D->SetTitle(Form("%s (%s)", hNoise2D->GetTitle(), pedestalFile.Data()));

  // ===| fill 1D histograms |===
  for (const auto& val : rocPedestal.getData()) {
    if (val > 0)
      hPedestal->Fill(val);
  }

  for (const auto& val : rocNoise.getData()) {
    if (val > 0)
      hNoise->Fill(val);
  }

  return {hPedestal2D, hNoise2D, hPedestal, hNoise};
}

void comparePedestalsAndNoise(const TString file1, const TString file2, int roc = 0)
{
  auto [hPedestal2D, hNoise2D, hPedestal, hNoise] = getNoiseAndPedestalHistogram(file1, roc);
  auto [hPedestal2D2, hNoise2D2, hPedestal2, hNoise2] = getNoiseAndPedestalHistogram(file2, roc);

  auto hPedestal2DDiff = new TH2F(*(TH2F*)hPedestal2D);
  auto hPedestalDiff = new TH1F(*(TH1F*)hPedestal);
  auto hNoise2DDiff = new TH2F(*(TH2F*)hNoise2D);
  auto hNoiseDiff = new TH1F(*(TH1F*)hNoise);

  hPedestal2DDiff->Add(hPedestal2D2, -1);
  hNoise2DDiff->Add(hNoise2D2, -1);
  hPedestalDiff->Add(hPedestal2, -1);
  hNoiseDiff->Add(hNoise2, -1);

  hPedestal2DDiff->SetTitle(Form("%s - %s", hPedestal2D->GetTitle(), hPedestal2D2->GetTitle()));
  hNoise2DDiff->SetTitle(Form("%s - %s", hNoise2D->GetTitle(), hNoise2D2->GetTitle()));
  hPedestalDiff->SetTitle(Form("%s - %s", hPedestal->GetTitle(), hPedestal2->GetTitle()));
  hNoiseDiff->SetTitle(Form("%s - %s", hNoise->GetTitle(), hNoise2->GetTitle()));

  // ===| draw histograms |===
  auto cPedestal = new TCanvas("cPedestal", "Pedestals");
  cPedestal->Divide(2, 2);
  cPedestal->cd(1);
  hPedestal->Draw();
  cPedestal->cd(2);
  hPedestal2->Draw();
  cPedestal->cd(3);
  hPedestalDiff->Draw();

  auto cNoise = new TCanvas("cNoise", "Noise");
  cNoise->Divide(2, 2);
  cNoise->cd(1);
  hNoise->Draw();
  cNoise->cd(2);
  hNoise2->Draw();
  cNoise->cd(3);
  hNoiseDiff->Draw();

  auto cPedestal2D = new TCanvas("cPedestal2D", "Pedestals2D");
  cPedestal2D->Divide(2, 2);
  cPedestal2D->cd(1);
  hPedestal2D->Draw("colz");
  cPedestal2D->cd(2);
  hPedestal2D2->Draw("colz");
  cPedestal2D->cd(3);
  hPedestal2DDiff->Draw("colz");

  auto cNoise2D = new TCanvas("cNoise2D", "Noise2D");
  cNoise2D->Divide(2, 2);
  cNoise2D->cd(1);
  hNoise2D->Draw("colz");
  cNoise2D->cd(2);
  hNoise2D2->Draw("colz");
  cNoise2D->cd(3);
  hNoise2DDiff->Draw("colz");
}
