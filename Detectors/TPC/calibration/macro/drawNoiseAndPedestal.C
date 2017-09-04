// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

void drawNoiseAndPedestal(TString pedestalFile)
{
  using namespace o2::TPC;
  TFile f(pedestalFile);
  gROOT->cd();

  // ===| load noise and pedestal from file |===
  CalDet<float> dummy;
  CalDet<float> *pedestal=nullptr, *noise=nullptr;
  f.GetObject("Pedestals", pedestal);
  f.GetObject("Noise", noise);
  const auto& rocPedestal = pedestal->getCalArray(0);
  const auto& rocNoise = noise->getCalArray(0);

  // ===| histograms for noise and pedestal |===
  auto hPedestal = new TH1F("hPedestal","Pedestal distribution;ADC value", 100, 50, 150);
  auto hNoise = new TH1F("hNoise","Noise distribution;ADC value", 100, 0, 5);
  auto hPedestal2D = Painter::getHistogram2D(rocPedestal);
  auto hNoise2D = Painter::getHistogram2D(rocNoise);

  // ===| fill 1D histograms |===
  for (const auto& val : rocPedestal.getData()) {
    if (val>0) hPedestal->Fill(val);
  }

  for (const auto& val : rocNoise.getData()) {
    if (val>0) hNoise->Fill(val);
  }

  // ===| draw histograms |===
  auto cPedestal=new TCanvas("cPedestal","Pedestals");
  hPedestal->Draw();

  auto cNoise=new TCanvas("cNoise","Noise");
  hNoise->Draw();

  auto cPedestal2D=new TCanvas("cPedestal2D","Pedestals2D");
  hPedestal2D->Draw("colz");

  auto cNoise2D=new TCanvas("cNoise2D","Noise2D");
  hNoise2D->Draw("colz");
}
