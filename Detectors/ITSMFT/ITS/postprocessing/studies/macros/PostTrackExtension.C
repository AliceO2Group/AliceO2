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
#include "TF1.h"
#include "TEfficiency.h"
#include "TMarker.h"
#include "TLegend.h"
#include "TTree.h"
#include "TLatex.h"

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
    {
      auto p = c->cd(1);
      p->SetGrid();
      auto h = p->DrawFrame(-.6, 0., .6, 9.);
      h->GetXaxis()->SetTitle("#frac{Q^{2}}{p_{T,TRK}}-#frac{Q^{2}}{p_{T,MC}}");
      h->GetYaxis()->SetTitle("n. counts");
      t->Draw("trk.getQ2Pt()-mcTrk.getQ2Pt()>>hPtNo(100,-.6,.6)", "isGood&&!isExtended", "HIST;SAME");
      auto hNo = (TH1F*)p->GetPrimitive("hPtNo");
      hNo->Scale(1.0 / hNo->Integral("width"));
      hNo->SetLineColor(kRed);
      auto fitNo = new TF1("fitNo", "gaus", -0.04, 0.04);
      hNo->Fit(fitNo, "QR");
      fitNo->SetLineColor(kRed);
      fitNo->Draw("SAME");
      auto textNo = new TLatex(-0.55, 8.2, Form("#mu = %.3f, #sigma = %.3f", fitNo->GetParameter(1), fitNo->GetParameter(2)));
      textNo->SetTextColor(kRed);
      textNo->SetNDC(false);
      textNo->SetTextSize(0.05);
      textNo->Draw();

      t->Draw("trk.getQ2Pt()-mcTrk.getQ2Pt()>>hPtYes(100,-.6,.6)", "isGood&&isExtended", "HIST;SAME");
      auto hYes = (TH1F*)p->GetPrimitive("hPtYes");
      hYes->Scale(1.0 / hYes->Integral("width"));
      hYes->SetLineColor(kBlue);
      auto fitYes = new TF1("fitYes", "gaus", -0.04, 0.04);
      hYes->Fit(fitYes, "QR");
      fitYes->SetLineColor(kBlue);
      fitYes->Draw("SAME");
      auto textYes = new TLatex(-0.55, 7, Form("#mu = %.4f, #sigma = %.4f", fitNo->GetParameter(1), fitNo->GetParameter(2)));
      textYes->SetTextColor(kBlue);
      textYes->SetNDC(false);
      textYes->SetTextSize(0.05);
      textYes->Draw();

      p->Modified();
      p->Update();
    }
    {
      auto p = c->cd(2);
      p->SetGrid();
      auto h = p->DrawFrame(-3, 0., 3, 2.);
      h->GetXaxis()->SetTitle("Y_{TRK}-Y_{MC}");
      h->GetYaxis()->SetTitle("n. counts");
      t->Draw("trk.getY()-mcTrk.getY()>>hYNo(100,-3,3)", "isGood&&!isExtended", "HIST;SAME");
      auto hNo = (TH1F*)p->GetPrimitive("hYNo");
      hNo->Scale(1.0 / hNo->Integral("width"));
      hNo->SetLineColor(kRed);
      auto fitNo = new TF1("fitNo", "gaus", -0.5, 0.5);
      hNo->Fit(fitNo, "QR");
      fitNo->SetLineColor(kRed);
      fitNo->Draw("SAME");
      auto textNo = new TLatex(-2, 1.7, Form("#mu = %.3f, #sigma = %.3f", fitNo->GetParameter(1), fitNo->GetParameter(2)));
      textNo->SetTextColor(kRed);
      textNo->SetNDC(false);
      textNo->SetTextSize(0.05);
      textNo->Draw();

      t->Draw("trk.getY()-mcTrk.getY()>>hYYes(100,-3,3)", "isGood&&isExtended", "HIST;SAME");
      auto hYes = (TH1F*)p->GetPrimitive("hYYes");
      hYes->Scale(1.0 / hYes->Integral("width"));
      hYes->SetLineColor(kBlue);
      auto fitYes = new TF1("fitYes", "gaus", -0.5, 0.5);
      hYes->Fit(fitYes, "QR");
      fitYes->SetLineColor(kBlue);
      fitYes->Draw("SAME");
      auto textYes = new TLatex(-2, 1.5, Form("#mu = %.4f, #sigma = %.4f", fitNo->GetParameter(1), fitNo->GetParameter(2)));
      textYes->SetTextColor(kBlue);
      textYes->SetNDC(false);
      textYes->SetTextSize(0.05);
      textYes->Draw();

      p->Modified();
      p->Update();
    }
    {
      auto p = c->cd(3);
      p->SetGrid();
      auto h = p->DrawFrame(-2, 0., 2, 4.2);
      h->GetXaxis()->SetTitle("Z_{TRK}-Z_{MC}");
      h->GetYaxis()->SetTitle("n. counts");
      t->Draw("trk.getZ()-mcTrk.getZ()>>hZNo(100,-2,2)", "isGood&&!isExtended", "HIST;SAME");
      auto hNo = (TH1F*)p->GetPrimitive("hZNo");
      hNo->Scale(1.0 / hNo->Integral("width"));
      hNo->SetLineColor(kRed);
      auto fitNo = new TF1("fitNo", "gaus", -0.2, 0.2);
      hNo->Fit(fitNo, "QR");
      fitNo->SetLineColor(kRed);
      fitNo->Draw("SAME");
      auto textNo = new TLatex(-1.7, 3.8, Form("#mu = %.3f, #sigma = %.3f", fitNo->GetParameter(1), fitNo->GetParameter(2)));
      textNo->SetTextColor(kRed);
      textNo->SetNDC(false);
      textNo->SetTextSize(0.05);
      textNo->Draw();

      t->Draw("trk.getZ()-mcTrk.getZ()>>hZYes(100,-2,2)", "isGood&&isExtended", "HIST;SAME");
      auto hYes = (TH1F*)p->GetPrimitive("hZYes");
      hYes->Scale(1.0 / hYes->Integral("width"));
      hYes->SetLineColor(kBlue);
      auto fitYes = new TF1("fitYes", "gaus", -0.2, 0.2);
      hYes->Fit(fitYes, "QR");
      fitYes->SetLineColor(kBlue);
      fitYes->Draw("SAME");
      auto textYes = new TLatex(-1.7, 3.5, Form("#mu = %.4f, #sigma = %.4f", fitNo->GetParameter(1), fitNo->GetParameter(2)));
      textYes->SetTextColor(kBlue);
      textYes->SetNDC(false);
      textYes->SetTextSize(0.05);
      textYes->Draw();

      p->Modified();
      p->Update();
    }
    {
      auto p = c->cd(4);
      p->SetGrid();
      auto h = p->DrawFrame(-0.02, 0., 0.02, 370.);
      h->GetXaxis()->SetTitle("TGL_{TRK}-TGL_{MC}");
      h->GetYaxis()->SetTitle("n. counts");
      t->Draw("trk.getTgl()-mcTrk.getTgl()>>hTglNo(100,-0.02,0.02)", "isGood&&!isExtended", "HIST;SAME");
      auto hNo = (TH1F*)p->GetPrimitive("hTglNo");
      hNo->Scale(1.0 / hNo->Integral("width"));
      hNo->SetLineColor(kRed);
      auto fitNo = new TF1("fitNo", "gaus", -0.003, 0.003);
      hNo->Fit(fitNo, "QR");
      fitNo->SetLineColor(kRed);
      fitNo->Draw("SAME");
      auto textNo = new TLatex(-0.018, 330, Form("#mu = %.3f, #sigma = %.3f", fitNo->GetParameter(1), fitNo->GetParameter(2)));
      textNo->SetTextColor(kRed);
      textNo->SetNDC(false);
      textNo->SetTextSize(0.05);
      textNo->Draw();

      t->Draw("trk.getTgl()-mcTrk.getTgl()>>hTglYes(100,-0.02,0.02)", "isGood&&isExtended", "HIST;SAME");
      auto hYes = (TH1F*)p->GetPrimitive("hTglYes");
      hYes->Scale(1.0 / hYes->Integral("width"));
      hYes->SetLineColor(kBlue);
      auto fitYes = new TF1("fitYes", "gaus", -0.003, 0.003);
      hYes->Fit(fitYes, "QR");
      fitYes->SetLineColor(kBlue);
      fitYes->Draw("SAME");
      auto textYes = new TLatex(-0.018, 310, Form("#mu = %.6f, #sigma = %.6f", fitNo->GetParameter(1), fitNo->GetParameter(2)));
      textYes->SetTextColor(kBlue);
      textYes->SetNDC(false);
      textYes->SetTextSize(0.05);
      textYes->Draw();

      p->Modified();
      p->Update();
    }
    {
      auto p = c->cd(5);
      p->SetGrid();
      auto h = p->DrawFrame(-0.08, 0., 0.08, 80.);
      h->GetXaxis()->SetTitle("SNP_{TRK}-SNP_{MC}");
      h->GetYaxis()->SetTitle("n. counts");
      t->Draw("trk.getSnp()-mcTrk.getSnp()>>hSnpNo(100,-0.08,0.08)", "isGood&&!isExtended", "HIST;SAME");
      auto hNo = (TH1F*)p->GetPrimitive("hSnpNo");
      hNo->Scale(1.0 / hNo->Integral("width"));
      hNo->SetLineColor(kRed);
      auto fitNo = new TF1("fitNo", "gaus", -0.03, 0.03);
      hNo->Fit(fitNo, "QR");
      fitNo->SetLineColor(kRed);
      fitNo->Draw("SAME");
      auto textNo = new TLatex(-0.07, 72, Form("#mu = %.3f, #sigma = %.3f", fitNo->GetParameter(1), fitNo->GetParameter(2)));
      textNo->SetTextColor(kRed);
      textNo->SetNDC(false);
      textNo->SetTextSize(0.05);
      textNo->Draw();

      t->Draw("trk.getSnp()-mcTrk.getSnp()>>hSnpYes(100,-0.08,0.08)", "isGood&&isExtended", "HIST;SAME");
      auto hYes = (TH1F*)p->GetPrimitive("hSnpYes");
      hYes->Scale(1.0 / hYes->Integral("width"));
      hYes->SetLineColor(kBlue);
      auto fitYes = new TF1("fitYes", "gaus", -0.03, 0.03);
      hYes->Fit(fitYes, "QR");
      fitYes->SetLineColor(kBlue);
      fitYes->Draw("SAME");
      auto textYes = new TLatex(-0.07, 66, Form("#mu = %.6f, #sigma = %.6f", fitNo->GetParameter(1), fitNo->GetParameter(2)));
      textYes->SetTextColor(kBlue);
      textYes->SetNDC(false);
      textYes->SetTextSize(0.05);
      textYes->Draw();

      p->Modified();
      p->Update();
    }
    {
      auto p = c->cd(6);
      auto legend = new TLegend(0.2, 0.2, 0.8, 0.8);
      legend->SetTextSize(0.06);
      legend->SetLineWidth(3);
      legend->SetHeader("GOOD tracks", "C");
      auto mBlue = new TMarker();
      mBlue->SetMarkerColor(kBlue);
      mBlue->SetMarkerSize(4);
      legend->AddEntry(mBlue, "extended", "p");
      auto mRed = new TMarker();
      mRed->SetMarkerColor(kRed);
      mRed->SetMarkerSize(4);
      legend->AddEntry(mRed, "normal", "p");
      legend->SetLineColor(kRed);
      legend->Draw();
    }
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
