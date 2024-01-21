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
//
// Original authors: M. Sitta, F. Grosa
// Author: M. Concas

#if !defined(__CLING__) || defined(__ROOTCLING__)

#include "DetectorsBase/GeometryManager.h"
#include "ITSBase/GeometryTGeo.h"
#include <TCanvas.h>
#include <TFile.h>
#include <TGeoManager.h>
#include <TH2F.h>
#include <TMath.h>
#include <TRandom.h>
#include <TSystem.h>
#include <TLegend.h>
#include <TStyle.h>
#include <TGeoMaterial.h>
#include <TGeoMedium.h>
#include <TGeoTube.h>
#include <TGeoVolume.h>
#include <TPaveText.h>
#include <TROOT.h>
#include <TText.h>
#include <THStack.h>

// #include "FairGeoParSet.h"     // for FairGeoParSet
#include <fairlogger/Logger.h> // for LOG, LOG_IF

#include <vector>
#include <string>
#include <algorithm>
#endif

constexpr int nBinsPhiScan = 90;
constexpr int nBinsEtaScan = 200;
constexpr int nBinsZvtxScan = 300;
constexpr float maxEtaScan = 8.;
constexpr int n = 1e4;      // testtracks
constexpr float len = 1000; // cm

void vacuumFormMaterial(TGeoMaterial* mat)
{
  constexpr double kAirA = 14.00674;
  constexpr double kAirZ = 7.;
  constexpr double kAirDensity = 0.0; // set for vacuum
  constexpr double kAirRadLen = std::numeric_limits<double>::max();
  // LOGP(info, "Vacuum forming {} ...", mat->GetName());
  // std::cout << "\t A = " << mat->GetA() << " -> " << kAirA << std::endl;
  // std::cout << "\t Z = " << mat->GetZ() << " -> " << kAirZ << std::endl;
  // std::cout << "\t Density = " << mat->GetDensity() << " -> " << kAirDensity << std::endl;
  // std::cout << "\t RadLen = " << mat->GetRadLen() << " -> " << kAirRadLen << std::endl;

  // Make this material air-like
  mat->SetA(kAirA);
  mat->SetZ(kAirZ);
  mat->SetDensity(kAirDensity);
  mat->SetRadLen(kAirRadLen);
}

//______________________________________________________________________
void ComputeMaterialBudget(double rmin, double rmax, double etaPos,
                           double phiMin, double phiMax, TH1F* xOverX0VsPhi,
                           TH1F* xOverX0VsEta, TH1F* xOverX0VsZvtx, TH2F* xOverX02d = nullptr)
{
  // Ancillary function to compute material budget between rmin and rmax

  TH1F* nParticlesVsPhi = new TH1F("", "", nBinsPhiScan, phiMin, phiMax);
  TH1F* nParticlesVsEta = new TH1F("", "", nBinsEtaScan, -maxEtaScan, maxEtaScan);
  TH1F* nParticlesVsZvtx = new TH1F("", "", nBinsZvtxScan, -len / 2, len / 2);
  TH2F* nParticlesVsPhiEta = new TH2F("", "", nBinsZvtxScan, -maxEtaScan, maxEtaScan, nBinsPhiScan, phiMax, phiMax);

  double x1, y1, z1, x2, y2, z2;

  for (int it = 0; it < n; it++) { // we simulate flat in phi and eta, from Zvtx=0

    // PHI VS ETA
    double phi = gRandom->Uniform(phiMin, phiMax);
    double eta = gRandom->Uniform(-maxEtaScan, maxEtaScan);
    double theta = TMath::ATan(TMath::Exp(-eta)) * 2;
    x1 = rmin * TMath::Cos(phi);
    y1 = rmin * TMath::Sin(phi);
    z1 = rmin / TMath::Tan(theta);
    x2 = rmax * TMath::Cos(phi);
    y2 = rmax * TMath::Sin(phi);
    z2 = rmax / TMath::Tan(theta);

    auto mparam = o2::base::GeometryManager::meanMaterialBudget(x1, y1, z1, x2, y2, z2);

    if (std::abs(eta) < etaPos) {
      xOverX0VsPhi->Fill(phi, mparam.meanX2X0 * 100);
      nParticlesVsPhi->Fill(phi);
    }
    xOverX0VsEta->Fill(eta, mparam.meanX2X0 * 100);
    nParticlesVsEta->Fill(eta);

    if (xOverX02d != 0) {
      xOverX02d->Fill(eta, phi, mparam.meanX2X0 * 100);
      nParticlesVsPhiEta->Fill(eta, phi);
    }
  }

  for (int it = 0; it < n; it++) { // then we simulate flat in Zvtx with eta = 0 and flat in phi

    double phi = TMath::Pi() / 2;
    double eta = 0.;
    double theta = TMath::ATan(TMath::Exp(-eta)) * 2;
    double z = gRandom->Uniform(-len / 2, len / 2);
    x1 = rmin * TMath::Cos(phi);
    y1 = rmin * TMath::Sin(phi);
    x2 = rmax * TMath::Cos(phi);
    y2 = rmax * TMath::Sin(phi);

    auto mparam = o2::base::GeometryManager::meanMaterialBudget(x1, y1, z, x2, y2, z);

    xOverX0VsZvtx->Fill(z, mparam.meanX2X0 * 100);
    nParticlesVsZvtx->Fill(z);
  }

  // normalization to number of particles in case of phi vs eta
  double theta = TMath::ATan(TMath::Exp(-etaPos / 2)) * 2;
  LOGP(info, "<η>={} -> Sin(θ) {}", etaPos / 2, TMath::Sin(theta));

  for (int ix = 1; ix <= nParticlesVsPhi->GetNbinsX(); ix++) {
    if (nParticlesVsPhi->GetBinContent(ix) > 0)
      xOverX0VsPhi->SetBinContent(ix, xOverX0VsPhi->GetBinContent(ix) / nParticlesVsPhi->GetBinContent(ix) * TMath::Sin(theta));
  }

  for (int ix = 1; ix <= nParticlesVsEta->GetNbinsX(); ix++) {
    if (nParticlesVsEta->GetBinContent(ix) > 0)
      xOverX0VsEta->SetBinContent(ix, xOverX0VsEta->GetBinContent(ix) / nParticlesVsEta->GetBinContent(ix));
  }

  for (int ix = 1; ix <= nParticlesVsZvtx->GetNbinsX(); ix++) {
    if (nParticlesVsZvtx->GetBinContent(ix) > 0)
      xOverX0VsZvtx->SetBinContent(ix, xOverX0VsZvtx->GetBinContent(ix) / nParticlesVsZvtx->GetBinContent(ix));
  }

  if (xOverX02d) {
    for (int ix = 1; ix <= nParticlesVsPhiEta->GetNbinsX(); ix++) {
      for (int iy = 1; iy <= nParticlesVsPhiEta->GetNbinsY(); iy++) {
        if (nParticlesVsPhiEta->GetBinContent(ix, iy) > 0)
          xOverX02d->SetBinContent(ix, iy, xOverX02d->GetBinContent(ix, iy) / nParticlesVsPhiEta->GetBinContent(ix, iy));
      }
    }
  }
}

std::vector<std::string> printMaterialDefinitions(TGeoManager* gman)
{
  std::vector<std::string> materialNames;
  TGeoMedium* med;
  TGeoMaterial* mat;
  char mediaName[50], matName[50], shortName[50];

  int nMedia = gman->GetListOfMedia()->GetEntries();

  LOGP(info, " =================== ALICE 3 Material Properties ================= ");
  LOGP(info, "    A      Z   d (g/cm3)  RadLen (cm)  IntLen (cm)\t Name\n");

  for (int i = 0; i < nMedia; i++) {
    med = (TGeoMedium*)(gman->GetListOfMedia()->At(i));
    mat = med->GetMaterial();
    LOGP(info, "{:5.1f} {:6.1f} {:8.3f} {:13.1f} {:11.1f}\t {}", mat->GetA(), mat->GetZ(), mat->GetDensity(), mat->GetRadLen(), mat->GetIntLen(), mat->GetName());

    std::vector<std::string> tokens;
    std::string matNameStr(mat->GetName());
    if (matNameStr.back() == '$') {
      matNameStr.pop_back();
    }
    size_t pos = 0;
    while ((pos = matNameStr.find("_")) != std::string::npos) {
      std::string part = matNameStr.substr(0, pos);
      tokens.push_back(part);
      matNameStr.erase(0, pos + 1);
    }
    std::transform(matNameStr.begin(), matNameStr.end(), matNameStr.begin(), ::toupper);
    tokens.push_back(matNameStr);
    if (tokens.back() == "NF") { // Manually manage air_NF
      continue;
    }
    if (std::find(materialNames.begin(), materialNames.end(), tokens.back()) == materialNames.end()) {
      materialNames.push_back(tokens.back());
    }
  }

  // print material names for debug
  for (auto& name : materialNames) {
    LOGP(info, "Unique material name: {}", name);
  }

  return materialNames;
}

void scanXX0(const float rmax = 200, const float rmin = 0.2, const std::string OnlyMat = "all", const string fileName = "o2sim_geometry.root", const string path = "./")
{
  gStyle->SetPadTopMargin(0.035);
  gStyle->SetPadRightMargin(0.035);
  gStyle->SetPadBottomMargin(0.14);
  gStyle->SetPadLeftMargin(0.14);
  gStyle->SetTitleOffset(1.4, "y");
  gStyle->SetPadTickX(1);
  gStyle->SetPadTickY(1);
  gStyle->SetOptStat(0);
  gStyle->SetOptTitle(0);
  gStyle->SetPalette(kSolar);

  double etaPos = 1.;

  TCanvas* canvStack = new TCanvas("canvStack", "canvStack", 2400, 800);
  canvStack->Divide(3, 1);

  TCanvas* canv = new TCanvas("canv", "canv", 2400, 800);
  canv->Divide(3, 1);

  TLegend* legVsPhi = new TLegend(0.25, 0.6, 0.85, 0.9);
  legVsPhi->SetFillColor(kWhite);
  legVsPhi->SetTextSize(0.045);
  legVsPhi->SetHeader(Form("ALICE 3, |#it{#eta}| < %0.f, #it{Z}_{vtx} = 0", etaPos));

  TLegend* legVsEta = new TLegend(0.25, 0.6, 0.85, 0.9);
  legVsEta->SetFillColor(kWhite);
  legVsEta->SetTextSize(0.045);
  legVsEta->SetHeader("ALICE 3, 0 < #it{#varphi} < #pi, #it{Z}_{vtx} = 0");

  TLegend* legVsZvtx = new TLegend(0.25, 0.6, 0.85, 0.9);
  legVsZvtx->SetFillColor(kWhite);
  legVsZvtx->SetTextSize(0.045);
  legVsZvtx->SetHeader("ALICE 3, #it{#varphi} = #pi/2, #it{#eta} = 0");

  auto* xOverX0VsPhiStack = new THStack("xOverX0VsPhi", "");
  auto* xOverX0VsEtaStack = new THStack("xOverX0VsEta", "");
  auto* xOverX0VsZvtxStack = new THStack("xOverX0VsZvtx", "");

  std::vector<TH1F*> xOverX0VsPhi;
  std::vector<TH1F*> xOverX0VsEta;
  std::vector<TH1F*> xOverX0VsZvtx;

  TGeoManager::Import((path + fileName).c_str());
  auto materials = printMaterialDefinitions(gGeoManager);

  const double phiMin = 0;
  const double phiMax = 2 * TMath::Pi();
  const double len = 1000.;
  std::vector<int> colors = {kAzure + 4, kRed + 1};

  // delete gGeoManager; // We re-import the geometry at each iteration
  int count = 2;
  auto cols = TColor::GetPalette();

  for (size_t iMaterial{0}; iMaterial < materials.size(); ++iMaterial) {
    if (OnlyMat != "all" && materials[iMaterial] != OnlyMat) {
      continue;
    }
    TGeoManager::Import((path + fileName).c_str());
    LOGP(info, " ********* Processing material: {} ********* ", materials[iMaterial]);
    auto nMedia = gGeoManager->GetListOfMedia()->GetEntries();
    for (int i = 0; i < nMedia; i++) {
      auto* med = (TGeoMedium*)(gGeoManager->GetListOfMedia()->At(i));
      auto* mat = med->GetMaterial();
      std::string matname{mat->GetName()};
      std::transform(matname.begin(), matname.end(), matname.begin(), ::toupper);
      if (matname.find(materials[iMaterial]) == std::string::npos) {
        vacuumFormMaterial(mat);
      } else {
        LOGP(info, "\t {} found as {} element.", materials[iMaterial], iMaterial);
      }
    }

    xOverX0VsPhi.emplace_back(new TH1F(Form("xOverX0VsPhi_step%zu", iMaterial), "", nBinsPhiScan, phiMin, phiMax));
    xOverX0VsEta.emplace_back(new TH1F(Form("xOverX0VsEta_step%zu", iMaterial), "", nBinsEtaScan, -maxEtaScan, maxEtaScan));
    xOverX0VsZvtx.emplace_back(new TH1F(Form("xOverX0VsZvtx_step%zu", iMaterial), "", nBinsZvtxScan, -len / 2, len / 2));

    ComputeMaterialBudget(rmin, rmax, etaPos, phiMin, phiMax, xOverX0VsPhi.back(), xOverX0VsEta.back(), xOverX0VsZvtx.back());

    double meanX0vsPhi = 0, meanX0vsEta = 0, meanX0vsZvtx = 0;
    for (int ix = 1; ix <= xOverX0VsPhi.back()->GetNbinsX(); ix++) {
      meanX0vsPhi += xOverX0VsPhi.back()->GetBinContent(ix);
    }
    meanX0vsPhi /= xOverX0VsPhi.back()->GetNbinsX();

    for (int ix = 1; ix <= xOverX0VsEta.back()->GetNbinsX(); ix++) {
      meanX0vsEta += xOverX0VsEta.back()->GetBinContent(ix);
    }
    meanX0vsEta /= xOverX0VsEta.back()->GetNbinsX();

    for (int ix = 1; ix <= xOverX0VsZvtx.back()->GetNbinsX(); ix++) {
      meanX0vsZvtx += xOverX0VsZvtx.back()->GetBinContent(ix);
    }
    meanX0vsZvtx /= xOverX0VsZvtx.back()->GetNbinsX();

    LOGP(info, "Mean X/X0 vs. phi: {}", meanX0vsPhi);
    LOGP(info, "Mean X/X0 vs. eta: {}", meanX0vsEta);
    LOGP(info, "Mean X/X0 vs. Zvtx: {}", meanX0vsZvtx);

    xOverX0VsPhi.back()->GetXaxis()->SetTitle("#it{#varphi} (rad)");
    xOverX0VsPhi.back()->GetXaxis()->SetTitleSize(0.05);
    xOverX0VsPhi.back()->GetXaxis()->SetLabelSize(0.045);
    xOverX0VsPhi.back()->GetYaxis()->SetTitle("#it{X}/#it{X}_{0} (%)");
    xOverX0VsPhi.back()->GetYaxis()->SetTitleSize(0.05);
    xOverX0VsPhi.back()->GetYaxis()->SetLabelSize(0.045);
    xOverX0VsPhi.back()->GetYaxis()->SetDecimals();
    xOverX0VsPhi.back()->SetFillColorAlpha(iMaterial + 2, 0.5);
    xOverX0VsPhi.back()->SetLineColor(iMaterial + 2);
    xOverX0VsPhi.back()->SetLineWidth(2);

    xOverX0VsEta.back()->GetXaxis()->SetTitle("#it{#eta}");
    xOverX0VsEta.back()->GetXaxis()->SetTitleSize(0.05);
    xOverX0VsEta.back()->GetXaxis()->SetLabelSize(0.045);
    xOverX0VsEta.back()->GetYaxis()->SetTitle("#it{X}/#it{X}_{0} (%)");
    xOverX0VsEta.back()->GetYaxis()->SetTitleSize(0.05);
    xOverX0VsEta.back()->GetYaxis()->SetLabelSize(0.045);
    xOverX0VsEta.back()->GetYaxis()->SetDecimals();
    xOverX0VsEta.back()->SetFillColorAlpha(iMaterial + 2, 0.5);
    xOverX0VsEta.back()->SetLineColor(iMaterial + 2);
    xOverX0VsEta.back()->SetLineWidth(2);

    xOverX0VsZvtx.back()->GetXaxis()->SetTitle("#it{Z}_{vtx} (cm)");
    xOverX0VsZvtx.back()->GetXaxis()->SetTitleSize(0.05);
    xOverX0VsZvtx.back()->GetXaxis()->SetLabelSize(0.045);
    xOverX0VsZvtx.back()->GetYaxis()->SetTitle("#it{X}/#it{X}_{0} (%)");
    xOverX0VsZvtx.back()->GetYaxis()->SetTitleSize(0.05);
    xOverX0VsZvtx.back()->GetYaxis()->SetLabelSize(0.045);
    xOverX0VsZvtx.back()->GetYaxis()->SetDecimals();
    xOverX0VsZvtx.back()->SetFillColorAlpha(iMaterial + 2, 0.5);
    xOverX0VsZvtx.back()->SetLineColor(iMaterial + 2);
    xOverX0VsZvtx.back()->SetLineWidth(2);

    if (xOverX0VsPhi.size() == 1) {
      legVsPhi->AddEntry("", Form("#LT #it{X}/#it{X}_{0} #GT = %0.3f %%", meanX0vsPhi), "");
      legVsEta->AddEntry("", Form("#LT #it{X}/#it{X}_{0} #GT = %0.3f %%", meanX0vsEta), "");
      legVsZvtx->AddEntry("", Form("#LT #it{X}/#it{X}_{0} #GT = %0.3f %%", meanX0vsZvtx), "");
    }
    legVsPhi->AddEntry(xOverX0VsPhi.back(), materials[iMaterial].c_str(), "f");
    legVsEta->AddEntry(xOverX0VsPhi.back(), materials[iMaterial].c_str(), "f");
    legVsZvtx->AddEntry(xOverX0VsZvtx.back(), materials[iMaterial].c_str(), "f");

    canv->cd(1)->SetGrid();
    if (xOverX0VsPhi.size() == 1) {
      xOverX0VsPhi.back()->SetMinimum(1.e-4);
      // xOverX0VsPhi.back()->SetMaximum(20.f);
      xOverX0VsPhi.back()->DrawCopy("HISTO");
      legVsPhi->Draw();
      xOverX0VsPhiStack->Add(xOverX0VsPhi.back());
    } else {
      xOverX0VsPhi.back()->DrawCopy("HISTO SAME");
      xOverX0VsPhiStack->Add(xOverX0VsPhi.back());
    }

    canv->cd(2)->SetGrid();
    if (xOverX0VsEta.size() == 1) {
      xOverX0VsEta.back()->SetMinimum(1.e-4);
      // xOverX0VsEta.back()->SetMaximum(60.f);
      xOverX0VsEta.back()->DrawCopy("HISTO");
      legVsEta->Draw();
      xOverX0VsEtaStack->Add(xOverX0VsEta.back());
    } else {
      xOverX0VsEta.back()->DrawCopy("HISTO SAME");
      xOverX0VsEtaStack->Add(xOverX0VsEta.back());
    }

    canv->cd(3)->SetGrid();
    if (xOverX0VsZvtx.size() == 1) {
      xOverX0VsZvtx.back()->SetMinimum(1.e-4);
      // xOverX0VsZvtx.back()->SetMaximum(120.f);
      xOverX0VsZvtx.back()->DrawCopy("HISTO");
      legVsZvtx->Draw();
      xOverX0VsZvtxStack->Add(xOverX0VsZvtx.back());
    } else {
      xOverX0VsZvtx.back()->DrawCopy("HISTO SAME");
      xOverX0VsZvtxStack->Add(xOverX0VsZvtx.back());
    }
    delete gGeoManager;
  }
  canvStack->cd(1)->SetGrid();
  xOverX0VsPhiStack->Draw("HISTO");
  canvStack->cd(2)->SetGrid();
  xOverX0VsEtaStack->Draw("HISTO");
  canvStack->cd(3)->SetGrid();
  xOverX0VsZvtxStack->Draw("HISTO");
  canvStack->BuildLegend(0.25, 0.6, 0.85, 0.9);

  canv->SaveAs("alice3_material_vsphietaz.pdf");
  canvStack->SaveAs("alice3_material_vsphietaz_stack.pdf");
}