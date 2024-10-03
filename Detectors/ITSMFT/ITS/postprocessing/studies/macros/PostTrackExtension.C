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
#include "TStyle.h"
#include "TFile.h"
#include "TError.h"
#include "TColor.h"
#include "TCanvas.h"
#include "TH1F.h"
#include "TEfficiency.h"
#include "TLegend.h"
#include "TTree.h"

#include <memory>
#include <array>
#include <format>
#endif

static constexpr std::array<uint8_t, 9> bitPatternsBefore{15, 30, 31, 60, 62, 63, 120, 124, 126};
static constexpr std::array<uint8_t, 16> bitPatternsAfter{31, 47, 61, 62, 63, 79, 94, 95, 111, 121, 122, 123, 124, 125, 126, 127};
static constexpr std::array<int, 16> patternColors = {
  kRed,        // Red
  kBlue,       // Blue
  kGreen,      // Green
  kMagenta,    // Magenta
  kCyan,       // Cyan
  kOrange,     // Orange
  kViolet,     // Violet
  kYellow,     // Yellow
  kPink,       // Pink
  kAzure,      // Azure
  kSpring,     // Spring Green
  kTeal,       // Teal
  kBlack,      // Black
  kGray,       // Gray
  kOrange + 7, // Light Orange
  kBlue - 9    // Light Blue
};

// Marker styles
static constexpr std::array<int, 16> patternMarkers = {
  20, // Full circle
  21, // Full square
  22, // Full triangle up
  23, // Full triangle down
  24, // Open circle
  25, // Open square
  26, // Open triangle up
  27, // Open cross
  28, // Star
  29, // Plus sign
  30, // Open diamond
  31, // Full diamond
  32, // Cross
  33, // Circle with cross
  34, // X sign
  35  // Double open cross
};

enum Labels : unsigned int {
  eAll = 0,
  eGood,
  eFake,
  eFakeBefore,
  eFakeAfter,
  eFakeMix,
  eTopGood,
  eBotGood,
  eMixGood,
  eTopFake,
  eBotFake,
  eMixFake,
  eN,
};
static const std::array<const char* const, eN> names{
  "ALL #frac{ext trks}{all trks}",
  "GOOD #frac{good ext trks}{all ext trks}",
  "FAKE #frac{fake trks}{all ext trks}",
  "FAKE BF #frac{fake bf trks}{fake ext trks}",
  "FAKE AF #frac{fake af trks}{fake ext trks}",
  "FAKE MIX #frac{fake mix trks}{fake ext trks}",
  // Good Top/Bot/Mix
  "TOP #frac{good top ext trks}{good ext trks}",
  "BOT #frac{good bot ext trks}{good ext trks}",
  "MIX #frac{good mix ext trks}{good ext trks}",
  // Fake Top/Bot/Mix
  "TOP #frac{fake top ext trks}{fake ext trks}",
  "BOT #frac{fake bot ext trks}{fake ext trks}",
  "MIX #frac{fake mix ext trks}{fake ext trks}",
};
static const std::array<EColor, eN> colors{kBlack, kGreen, kRed, kCyan, kYellow, kAzure,
                                           // Good Top/Bot/Mix
                                           kBlue, kOrange, kPink,
                                           // Fake Top/Bot/Mix
                                           kBlue, kOrange, kPink};
static const std::array<int, eN> markers{20, 21, 22, 23, 27, 28,
                                         // Good Top/Bot/Mix
                                         29, 33, 39,
                                         // Fake Top/Bot/Mix
                                         29, 33, 39};
static const char* const texPtX = "#it{p}_{T} (GeV/#it{c})";
static const char* const texEff = "Efficiency";

void setStyle();
TEfficiency* makeEff(TFile*, const char* num, const char* den);

template <class T>
void style(T* t, Labels lab, TLegend* leg = nullptr)
{
  t->SetMarkerStyle(markers[lab]);
  t->SetMarkerColor(colors[lab]);
  t->SetLineColor(colors[lab]);
  if (leg) {
    leg->AddEntry(t, names[lab]);
  }
}

template <class T>
void stylePattern(T* t, int i, TLegend* leg = nullptr, const char* name = nullptr)
{
  t->SetMarkerStyle(patternMarkers[i]);
  t->SetMarkerColor(patternColors[i]);
  t->SetLineColor(patternColors[i]);
  if (leg) {
    leg->AddEntry(t, name);
  }
}

void PostTrackExtension(const char* fileName = "TrackExtensionStudy.root")
{
  setStyle();

  std::unique_ptr<TFile> fIn{TFile::Open(fileName, "READ")};
  if (!fIn || fIn->IsZombie()) {
    Error("", "Cannot open file %s", fileName);
    return;
  }

  { // Purity & Fake-Rate
    auto c = new TCanvas("cPFR", "", 800, 600);
    auto h = c->DrawFrame(0.05, 0.0, 10., 1.05);
    h->GetXaxis()->SetTitle(texPtX);
    h->GetYaxis()->SetTitle(texEff);
    auto leg = new TLegend(0.35, 0.35, 0.7, 0.7);
    auto eff = fIn->Get<TEfficiency>("eExtension");
    style(eff, eAll, leg);
    eff->Draw("same");
    auto effPurity = fIn->Get<TEfficiency>("eExtensionPurity");
    style(effPurity, eGood, leg);
    effPurity->Draw("same");
    auto effFake = fIn->Get<TEfficiency>("eExtensionFake");
    style(effFake, eFake, leg);
    effFake->Draw("same");
    leg->Draw();
    gPad->SetLogx();
    gPad->SetGrid();
    c->SaveAs("trkExt_purity_fake.pdf");
  }

  { // FAKE-Rate composition
    auto c = new TCanvas("cFR", "", 800, 600);
    auto h = c->DrawFrame(0.05, 0.0, 10., 1.05);
    h->GetXaxis()->SetTitle(texPtX);
    h->GetYaxis()->SetTitle(texEff);
    auto leg = new TLegend(0.35, 0.35, 0.7, 0.7);
    auto effFake = fIn->Get<TEfficiency>("eExtensionFake");
    style(effFake, eFake, leg);
    effFake->Draw("same");
    auto effFakeBf = fIn->Get<TEfficiency>("eExtensionFakeBefore");
    style(effFakeBf, eFakeBefore, leg);
    effFakeBf->Draw("same");
    auto effFakeAf = fIn->Get<TEfficiency>("eExtensionFakeAfter");
    style(effFakeAf, eFakeAfter, leg);
    effFakeAf->Draw("same");
    auto effFakeMi = fIn->Get<TEfficiency>("eExtensionFakeMix");
    style(effFakeMi, eFakeMix, leg);
    effFakeMi->Draw("same");
    leg->Draw();
    gPad->SetLogx();
    gPad->SetGrid();
    c->SaveAs("trkExt_fake.pdf");
  }

  { // GOOD Top/Bot/Mix Purity composition
    auto c = new TCanvas("cGC", "", 800, 600);
    auto h = c->DrawFrame(0.05, 0.0, 10., 1.05);
    h->GetXaxis()->SetTitle(texPtX);
    h->GetYaxis()->SetTitle(texEff);
    auto leg = new TLegend(0.35, 0.35, 0.7, 0.7);
    auto effTop = makeEff(fIn.get(), "eExtensionTopPurity", "eExtensionPurity");
    style(effTop, eTopGood, leg);
    effTop->Draw("same");
    auto effBot = makeEff(fIn.get(), "eExtensionBotPurity", "eExtensionPurity");
    style(effBot, eBotGood, leg);
    effBot->Draw("same");
    auto effMix = makeEff(fIn.get(), "eExtensionMixPurity", "eExtensionPurity");
    style(effMix, eMixGood, leg);
    effMix->Draw("same");
    leg->Draw();
    gPad->SetLogx();
    gPad->SetGrid();
    c->SaveAs("trkExt_good_comp.pdf");
  }

  { // FAKE Top/Bot/Mix composition
    auto c = new TCanvas("cFC", "", 800, 600);
    auto h = c->DrawFrame(0.05, 0.0, 10., 1.05);
    h->GetXaxis()->SetTitle(texPtX);
    h->GetYaxis()->SetTitle(texEff);
    auto leg = new TLegend(0.35, 0.35, 0.7, 0.7);
    auto effTop = fIn->Get<TEfficiency>("eExtensionTopFake");
    style(effTop, eTopFake, leg);
    effTop->Draw("same");
    auto effBot = fIn->Get<TEfficiency>("eExtensionBotFake");
    style(effBot, eBotFake, leg);
    effBot->Draw("same");
    auto effMix = fIn->Get<TEfficiency>("eExtensionMixFake");
    style(effMix, eMixFake, leg);
    effMix->Draw("same");
    leg->Draw();
    gPad->SetLogx();
    gPad->SetGrid();
    c->SaveAs("trkExt_fake_comp.pdf");
  }

  { // Good Patterns
    auto c = new TCanvas("cPatGood", "", 3 * 800, 3 * 600);
    c->Divide(3, 3);
    for (int i{0}; i < (int)bitPatternsBefore.size(); ++i) {
      auto p = c->cd(i + 1);
      auto h = p->DrawFrame(0.05, 0.0, 10., 1.05);
      h->GetXaxis()->SetTitle(texPtX);
      h->GetYaxis()->SetTitle(texEff);
      auto leg = new TLegend(0.35, 0.35, 0.7, 0.7);
      leg->SetNColumns(4);
      leg->SetHeader(std::format("BEFORE={:07b} GOOD Pattern AFTER/BEFORE", bitPatternsBefore[i]).c_str());
      for (int j{0}; j < (int)bitPatternsAfter.size(); ++j) {
        auto eff = fIn->Get<TEfficiency>(std::format("eExtensionPatternGood_{:07b}_{:07b}", bitPatternsBefore[i], bitPatternsAfter[j]).c_str());
        stylePattern(eff, j, leg, std::format("{:07b}", bitPatternsAfter[j]).c_str());
        eff->Draw("same");
      }
      leg->Draw();
      p->SetLogx();
      p->SetGrid();
    }
    c->SaveAs("trkExt_good_pattern_comp.pdf");
  }

  { // Fake Patterns
    auto c = new TCanvas("cPatFake", "", 3 * 800, 3 * 600);
    c->Divide(3, 3);
    for (int i{0}; i < (int)bitPatternsBefore.size(); ++i) {
      auto p = c->cd(i + 1);
      auto h = p->DrawFrame(0.05, 0.0, 10., 1.05);
      h->GetXaxis()->SetTitle(texPtX);
      h->GetYaxis()->SetTitle(texEff);
      auto leg = new TLegend(0.35, 0.35, 0.7, 0.7);
      leg->SetNColumns(4);
      leg->SetHeader(std::format("BEFORE={:07b} FAKE Pattern AFTER/BEFORE", bitPatternsBefore[i]).c_str());
      for (int j{0}; j < (int)bitPatternsAfter.size(); ++j) {
        auto eff = fIn->Get<TEfficiency>(std::format("eExtensionPatternFake_{:07b}_{:07b}", bitPatternsBefore[i], bitPatternsAfter[j]).c_str());
        stylePattern(eff, j, leg, std::format("{:07b}", bitPatternsAfter[j]).c_str());
        eff->Draw("same");
      }
      leg->Draw();
      p->SetLogx();
      p->SetGrid();
    }
    c->SaveAs("trkExt_fake_pattern_comp.pdf");
  }

  { // Kinematic variables
    auto t = fIn->Get<TTree>("tree");
    auto c = new TCanvas("cKG", "", 800, 600);
    c->Divide(3, 2);
    auto p = c->cd(1);
    p->SetGrid();
    auto h = p->DrawFrame(-.5, 0., .5, 30.);
    h->GetXaxis()->SetTitle("#it{p}_{T,TRK}-#it{p}_{T,MC}");
    h->GetYaxis()->SetTitle("n. counts");
    t->Draw("trk.getPt()-mcTrk.getPt()>>hPtNo(100,-.5,.5)", "isGood&&!isExtended", "HIST;SAME");
    auto htemp = (TH1F*)p->GetPrimitive("hPtNo");
    htemp->Scale(1.0 / htemp->Integral("width"));
    htemp->SetLineColor(kRed);
    t->Draw("trk.getPt()-mcTrk.getPt()>>hPtYes(100,-.5,.5)", "isGood&&isExtended", "HIST;SAME");
    htemp = (TH1F*)p->GetPrimitive("hPtYes");
    htemp->Scale(1.0 / htemp->Integral("width"));
    htemp->SetLineColor(kBlue);
    p->Modified();
    p->Update();
    c->SaveAs("trkExt_kinematics.pdf");
  }
}

void setStyle()
{
  gStyle->Reset("Plain");
  gStyle->SetOptTitle(0);
  gStyle->SetOptStat(0);
  gStyle->SetPalette(kRainbow);
  gStyle->SetCanvasColor(10);
  gStyle->SetCanvasBorderMode(0);
  gStyle->SetFrameLineWidth(1);
  gStyle->SetFrameFillColor(kWhite);
  gStyle->SetPadColor(10);
  gStyle->SetPadTickX(1);
  gStyle->SetPadTickY(1);
  gStyle->SetPadBottomMargin(0.15);
  gStyle->SetPadLeftMargin(0.15);
  gStyle->SetHistLineWidth(1);
  gStyle->SetHistLineColor(kRed);
  gStyle->SetFuncWidth(2);
  gStyle->SetFuncColor(kGreen);
  gStyle->SetLineWidth(2);
  gStyle->SetLabelSize(0.045, "xyz");
  gStyle->SetLabelOffset(0.01, "y");
  gStyle->SetLabelOffset(0.01, "x");
  gStyle->SetLabelColor(kBlack, "xyz");
  gStyle->SetTitleSize(0.05, "xyz");
  gStyle->SetTitleOffset(1.25, "y");
  gStyle->SetTitleOffset(1.2, "x");
  gStyle->SetTitleFillColor(kWhite);
  gStyle->SetTextSizePixels(26);
  gStyle->SetTextFont(42);
  gStyle->SetTickLength(0.04, "X");
  gStyle->SetTickLength(0.04, "Y");
  gStyle->SetLegendBorderSize(0);
  gStyle->SetLegendFillColor(kWhite);
  gStyle->SetFillColor(kWhite);
  gStyle->SetLegendFont(42);
}

TEfficiency* makeEff(TFile* fIn, const char* num, const char* den)
{
  auto h1 = fIn->Get<TEfficiency>(num)->GetPassedHistogram();
  auto h2 = fIn->Get<TEfficiency>(den)->GetPassedHistogram();
  auto e = new TEfficiency(*h1, *h2);
  return e;
}
