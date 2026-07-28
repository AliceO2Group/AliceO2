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

#include <fmt/format.h>
#include <string>
#include <algorithm>
#include <filesystem>
#include "TChain.h"
#include "TMath.h"
#include "TCanvas.h"
#include "THStack.h"
#include "TLegend.h"
#include "TProfile.h"
#include "TProfile2D.h"
#include "TObjArray.h"
#include "TStyle.h"
#include "TSystem.h"
#include "TLatex.h"
#include "TPaletteAxis.h"
#include "TPCBase/Utils.h"
#include "TPCBaseRecSim/Painter.h"
#include "CommonUtils/StringUtils.h"
#include "Framework/Logger.h"
#include "SpacePoints/TrackResiduals.h"

std::string getString(TList* l, const std::string& name, std::string defaultVal = "0");
int getNbins(const std::string& name, int defaultVal = 20, const char delim = ',');
void setStyle(TH1* hist, int idir);
void setSizes(TAxis* axis, float titleSize, float titleOffset, float labelSize);

using namespace o2::tpc;
namespace fs = std::filesystem;

void voxResQA(std::string inFilesCmd, std::string outFileNameAdd = "", bool drawErrors = false, bool useSmoothed = true, int z2xBinSel = -1)
{
  gStyle->SetOptStat(0);

  TObjArray arrCanvases1D;
  arrCanvases1D.SetName("voxResQA");
  TObjArray arrCanvases2D;
  arrCanvases2D.SetName("voxResQA");

  auto hsdXvsRowA = new THStack;
  auto hsdXvsRowC = new THStack;
  auto hsdZvsRowA = new THStack;
  auto hsdZvsRowC = new THStack;

  const std::string dXtitle = useSmoothed ? "dXS" : "dX";
  const std::string dYtitle = useSmoothed ? "dYS" : "dY";
  const std::string dZtitle = useSmoothed ? "dZS" : "dZ";

  const std::string sFiles(gSystem->GetFromPipe(inFilesCmd.data()));
  const auto arrFiles = o2::utils::Str::tokenize(sFiles, '\n');
  if (arrFiles.size() == 0) {
    return;
  }
  int idir = 0;
  for (const auto& inFile : arrFiles) {
    const fs::path inPath(inFile);
    const std::string fileTitle(std::string(inPath.stem()).substr(17));
    auto fileIDName = fileTitle;
    std::replace(fileIDName.begin(), fileIDName.end(), '.', '_');
    std::replace(fileIDName.begin(), fileIDName.end(), '-', '_');
    TChain cVoxRes("voxResTree");
    cVoxRes.AddFile(inFile.data());

    if (!cVoxRes.GetBranch("voxRes")) {
      LOGP(error, "Could not find branch voxRes in file '{}'", inFile);
      continue;
    }

    o2::tpc::TrackResiduals::VoxRes* vox{nullptr};
    cVoxRes.SetBranchAddress("voxRes", &vox);

    // retrieve binning
    // OBJ: TNamed	meanIDC	0.295421
    // OBJ: TNamed	meanCTP	118553.744497
    // OBJ: TNamed	y2xBinning	20
    // OBJ: TNamed	z2xBinning	20

    cVoxRes.GetEntry(0);
    const auto userInfo = cVoxRes.GetTree()->GetUserInfo();
    userInfo->Print();
    const auto y2xBinning = getString(userInfo, "y2xBinning");
    const auto z2xBinning = getString(userInfo, "z2xBinning");
    const auto meanIDC = std::stof(getString(userInfo, "meanIDC"));
    const auto meanCTP = std::stof(getString(userInfo, "meanCTP"));

    const int nbinsY2X = getNbins(y2xBinning);
    const int nBinsSector = nbinsY2X * 36;
    const int nbinsZ2X = getNbins(z2xBinning);

    // ===| 1D histograms |=====================================================
    TObjArray arrHists1D;
    auto hEntriesDist = new TH1F(("hEntriesDist" + fileIDName).data(), (fileTitle + ";#entries").data(), 500, 0, 5000);
    arrHists1D.Add(hEntriesDist);
    auto hdXDist = new TH1F(("hdXDist" + fileIDName).data(), (fileTitle + ";" + dXtitle + " (cm)").data(), 100, -10, 20);
    arrHists1D.Add(hdXDist);
    auto hdYDist = new TH1F(("hdYDist" + fileIDName).data(), (fileTitle + ";" + dYtitle + " (cm)").data(), 100, -10, 10);
    arrHists1D.Add(hdYDist);
    auto hdZDist = new TH1F(("hdZDist" + fileIDName).data(), (fileTitle + ";" + dZtitle + " (cm)").data(), 100, -10, 10);
    arrHists1D.Add(hdZDist);
    auto hdYSigmaMAD = new TProfile(("hdYSigmaMAD" + fileIDName).data(), (fileTitle + ";sector;<#sigma_{MAD}(dY)> (cm)").data(), nBinsSector, 0, nBinsSector);
    arrHists1D.Add(hdYSigmaMAD);

    auto hdXvsRowA = new TProfile(("hdXvsRowA" + fileIDName).data(), (fileTitle + " (A-Side);row;<" + dXtitle + "> (cm) z2xBin 0").data(), 152, 0, 152);
    hsdXvsRowA->Add(hdXvsRowA);
    hsdXvsRowA->SetTitle("(A-Side);row;<dX> (cm) z2xBin 0");
    setStyle(hdXvsRowA, idir);

    auto hdXvsRowC = new TProfile(("hdXvsRowC" + fileIDName).data(), (fileTitle + " (C-Side);row;<" + dXtitle + "> (cm) z2xBin 0").data(), 152, 0, 152);
    hsdXvsRowC->Add(hdXvsRowC);
    hsdXvsRowC->SetTitle("(C-Side);row;<dX> (cm) z2xBin 0");
    setStyle(hdXvsRowC, idir);

    auto hdZvsRowA = new TProfile(("hdZvsRowA" + fileIDName).data(), (fileTitle + " (A-Side);row;<" + dZtitle + "> (cm) z2xBin 0").data(), 152, 0, 152);
    hsdZvsRowA->Add(hdZvsRowA);
    hsdZvsRowA->SetTitle("(A-Side);row;<dZ> (cm) z2xBin 0");
    setStyle(hdZvsRowA, idir);

    auto hdZvsRowC = new TProfile(("hdZvsRowC" + fileIDName).data(), (fileTitle + " (C-Side);row;<" + dZtitle + "> (cm) z2xBin 0").data(), 152, 0, 152);
    hsdZvsRowC->Add(hdZvsRowC);
    hsdZvsRowC->SetTitle(("(C-Side);row;" + dZtitle + " (cm) z2xBin 0").data());
    setStyle(hdZvsRowC, idir);

    auto hdZvsZ2X = new TProfile(("hdZvsZ2X" + fileIDName).data(), (fileTitle + ";z2x-bin;<" + dZtitle + "> (cm) row 80").data(), 2 * nbinsZ2X, -nbinsZ2X, nbinsZ2X);
    arrHists1D.Add(hdZvsZ2X);

    // ===| 2D histograms |=====================================================
    TObjArray arrHists2D;
    auto hMeanEntries = new TProfile2D(("hMeanEntries_" + fileIDName).data(), (fileTitle + ";sector;row;<entries>").data(), nBinsSector, 0, nBinsSector, 152, 0, 152);
    arrHists2D.Add(hMeanEntries);
    auto hMeanEntrieszRow = new TProfile2D(("hMeanEntrieszRow" + fileIDName).data(), (fileTitle + ";z-bin;row;<entries> (cm)").data(), 2 * nbinsZ2X, -nbinsZ2X, nbinsZ2X, 152, 0, 152);
    arrHists2D.Add(hMeanEntrieszRow);
    auto hSigmaMADSecRow = new TProfile2D(("hSigmaMADSecRow" + fileIDName).data(), (fileTitle + ";sector;row;<#sigma_{MAD}(dY)> (cm)").data(), nBinsSector, 0, nBinsSector, 152, 0, 152);
    arrHists2D.Add(hSigmaMADSecRow);
    auto hdXSecRow = new TProfile2D(("hdXSecRow" + fileIDName).data(), (fileTitle + ";sector;row;<" + dXtitle + "> (cm)").data(), nBinsSector, 0, nBinsSector, 152, 0, 152);
    arrHists2D.Add(hdXSecRow);
    auto hdYSecRow = new TProfile2D(("hdYSecRow" + fileIDName).data(), (fileTitle + ";sector;row;<" + dYtitle + "> (cm)").data(), nBinsSector, 0, nBinsSector, 152, 0, 152);
    arrHists2D.Add(hdYSecRow);
    auto hdZzRow = new TProfile2D(("hdZzRow" + fileIDName).data(), (fileTitle + ";z2x-bin;row;<" + dZtitle + "> (cm)").data(), 2 * nbinsZ2X, -nbinsZ2X, nbinsZ2X, 152, 0, 152);
    arrHists2D.Add(hdZzRow);

    TProfile2D* hdXESecRow = nullptr;
    TProfile2D* hdYESecRow = nullptr;
    TProfile2D* hdZEzRow = nullptr;
    if (drawErrors) {
      hdXESecRow = new TProfile2D(("hdXESecRow" + fileIDName).data(), (fileTitle + ";sector;row;<dXE> (cm)").data(), nBinsSector, 0, nBinsSector, 152, 0, 152);
      arrHists2D.Add(hdXESecRow);
      hdYESecRow = new TProfile2D(("hdYESecRow" + fileIDName).data(), (fileTitle + ";sector;row;<dYE> (cm)").data(), nBinsSector, 0, nBinsSector, 152, 0, 152);
      arrHists2D.Add(hdYESecRow);
      hdZEzRow = new TProfile2D(("hdZEzRow" + fileIDName).data(), (fileTitle + ";z-bin;row;<dZE> (cm)").data(), 2 * nbinsZ2X, -nbinsZ2X, nbinsZ2X, 152, 0, 152);
      arrHists2D.Add(hdZEzRow);
    }

    /*
     OBJ: TNamed	z2xBin	bvox[0]
   OBJ: TNamed	y2xBin	bvox[1]
   OBJ: TNamed	xBin	bvox[2]
   OBJ: TNamed	z2xAV	stat[0]
   OBJ: TNamed	y2xAV	stat[1]
   OBJ: TNamed	xAV	stat[2]
   OBJ: TNamed	fsector	bsec+0.5+9.*(y2xAV)/pi
   OBJ: TNamed	phi	(bsec%18+0.5+9.*(stat[1])/pi)/9*pi
   OBJ: TNamed	r	stat[2]
   OBJ: TNamed	z	z2xAV*xAV
   OBJ: TNamed	dX	D[0]
   OBJ: TNamed	dY	D[1]
   OBJ: TNamed	dZ	D[2]
   OBJ: TNamed	dXS	DS[0]
   OBJ: TNamed	dYS	DS[1]
   OBJ: TNamed	dZS	DS[2]
   OBJ: TNamed	dXE	E[0]
   OBJ: TNamed	dYE	E[1]
   OBJ: TNamed	dZE	E[2]
   OBJ: TNamed	voxelIndex	xBin + 152 * (y2xBin + 20 * z2xBin) + 60800 * bsec
   OBJ: TNamed	entries	stat[3]
   OBJ: TNamed	fitOK	(flags & 1) == 1
   OBJ: TNamed	dispOK	(flags & 2) == 2
   OBJ: TNamed	smtOK	(flags & 4) == 4
   OBJ: TNamed	masked	(flags & 128) == 128
  */

    float z2xAV = 0;

    int isNaNdX = 0;
    int isNaNdY = 0;
    int isNaNdZ = 0;
    int isNaNdXS = 0;
    int isNaNdYS = 0;
    int isNaNdZS = 0;

    for (Long64_t iEntry = 0; iEntry < cVoxRes.GetEntries(); ++iEntry) {
      cVoxRes.GetEntry(iEntry);
      const auto y2xBin = vox->bvox[1];
      const auto z2xBin = vox->bvox[0];
      const auto xBin = vox->bvox[2];
      const auto bsec = vox->bsec;
      const auto entries = vox->stat[3];
      const auto dX = vox->D[0];
      const auto dY = vox->D[1];
      const auto dZ = vox->D[2];
      const auto dXS = vox->DS[0];
      const auto dYS = vox->DS[1];
      const auto dZS = vox->DS[2];
      const auto dXE = vox->E[0];
      const auto dYE = vox->E[1];
      const auto dZE = vox->E[2];
      const auto sectorFine = y2xBin + bsec * nbinsY2X;
      const auto z2xBinSides = (z2xBin + 0.5) * (1 - 2 * (bsec > 17));
      const auto z = vox->stat[0] * vox->stat[2];

      const auto dXdraw = useSmoothed ? dXS : dX;
      const auto dYdraw = useSmoothed ? dYS : dY;
      const auto dZdraw = useSmoothed ? dZS : dZ;

      isNaNdX += TMath::IsNaN(dX);
      isNaNdY += TMath::IsNaN(dY);
      isNaNdZ += TMath::IsNaN(dZ);
      isNaNdXS += TMath::IsNaN(dXS);
      isNaNdYS += TMath::IsNaN(dYS);
      isNaNdZS += TMath::IsNaN(dZS);

      // only fill values in the acceptance
      if (std::abs(z) < 248) {
        if (z2xBinSel < 0 || z2xBinSel == z2xBin) {
          if (z2xAV == 0) {
            z2xAV = vox->stat[0];
          }
          hMeanEntries->Fill(sectorFine, xBin, entries);
          hdXSecRow->Fill(sectorFine, xBin, dXdraw);
          hdYSecRow->Fill(sectorFine, xBin, dYdraw);
          hSigmaMADSecRow->Fill(sectorFine, xBin, vox->dYSigMAD);
          hdYSigmaMAD->Fill(sectorFine, vox->dYSigMAD);
        }
      }

      hMeanEntrieszRow->Fill(z2xBinSides, xBin, entries);
      hdZzRow->Fill(z2xBinSides, xBin, dZdraw);
      if (drawErrors && (z2xBinSel < 0 || z2xBinSel == z2xBin)) {
        hdXESecRow->Fill(sectorFine, xBin, dXE);
        hdYESecRow->Fill(sectorFine, xBin, dYE);
        hdZEzRow->Fill(z2xBinSides, xBin, dZE);
      }

      hEntriesDist->Fill(entries);
      hdXDist->Fill(dXdraw);
      hdYDist->Fill(dYdraw);
      hdZDist->Fill(dZdraw);
      if (xBin >= 79 && xBin <= 81) {
        hdZvsZ2X->Fill(z2xBinSides, dZdraw);
      }

      if (z2xBin == 0) {
        if (bsec < 18) {
          hdXvsRowA->Fill(xBin, dXdraw);
          hdZvsRowA->Fill(xBin, dZdraw);
        } else {
          hdXvsRowC->Fill(xBin, dXdraw);
          hdZvsRowC->Fill(xBin, dZdraw);
        }
      }
    }

    std::vector<TH1*> hXadjust{hMeanEntries, hdXSecRow, hdXESecRow, hdYSecRow, hdYESecRow};
    for (auto h : hXadjust) {
      if (!h) {
        continue;
      }
      h->GetXaxis()->SetLimits(0, 36);
      // 36 exact divisions means a label for every single sector -- unreadable at this pad size, they
      // overlap into an illegible block. 12 (label every 3 sectors) still shows the A-/C-side structure
      // without the collision.
      h->GetXaxis()->SetNdivisions(12, false);
    }

    // hEntriesDist's fixed 0-5000 range is mostly empty for real data (per-voxel entry counts rarely get
    // anywhere near 5000) -- zoom to just past the last populated bin instead of showing mostly blank
    // axis.
    if (hEntriesDist->GetEntries() > 0) {
      const int lastBin = hEntriesDist->FindLastBinAbove(0);
      if (lastBin > 0) {
        hEntriesDist->GetXaxis()->SetRangeUser(0, hEntriesDist->GetXaxis()->GetBinUpEdge(lastBin) * 1.1);
      }
    }

    // ===| output canvases |===================================================
    //
    auto c1D = new TCanvas(("c1D_" + fileIDName + outFileNameAdd).data(), fileTitle.data(), 1500, 900);
    arrCanvases1D.Add(c1D);

    int ipad = 1;
    c1D->DivideSquare(arrHists1D.GetEntries());

    for (auto o : arrHists1D) {
      c1D->cd(ipad++);
      o->Draw();
      const std::string name(o->GetName());
      if (o->IsA() != TProfile::Class()) {
        gPad->SetLogy();
      }
      // Without this, saveCanvas()'s plain c.SaveAs() (Detectors/TPC/base/src/Utils.cxx) can export a
      // pad before its log-scale range is actually recomputed in batch mode -- confirmed real: the
      // stored TCanvas in voxResQA*_1D.root has all real data (reopening and redrawing it interactively
      // shows every panel fine), but the direct PNG export came out blank. The c2D loop below already
      // does this after its own Draw() calls; c1D's was missing it.
      gPad->Modified();
      gPad->Update();
    }

    auto c2D = new TCanvas(("c2D_" + fileIDName + outFileNameAdd).data(), fileTitle.data(), 1500, 900);
    arrCanvases2D.Add(c2D);

    ipad = 1;
    c2D->DivideSquare(arrHists2D.GetEntries());

    TLatex l;
    l.SetTextFont(42);
    // Without this, DrawLatex's (x,y) below are interpreted in the pad's USER (data-axis) coordinates,
    // not normalized pad fractions -- 0.75/0.85 then lands almost at the frame's bottom-left corner
    // (sector~0.75 of 36, row~0.85 of 152), overlapping the plotted content instead of sitting clear of
    // it. SetNDC() makes the coordinates pad-fraction-based, and the y moved down (below the frame,
    // rather than "0.85" which was never actually near the top).
    l.SetNDC();

    for (auto o : arrHists2D) {
      auto h = static_cast<TH1*>(o);
      c2D->cd(ipad++);
      h->Draw("colz");
      const std::string name(h->GetName());
      if (name.find("hdZzRow") == 0) {
        l.DrawLatex(0.4, 0.02, "y2xBin averaged");
      } else {
        if (z2xBinSel >= 0) {
          l.DrawLatex(0.4, 0.02, fmt::format("z2xBin = {} ({})", z2xBinSel, z2xAV).data());
        } else {
          l.DrawLatex(0.4, 0.02, "z2xBin averaged");
        }
      }
      gPad->Modified();
      gPad->Update();

      auto palette = (TPaletteAxis*)h->GetListOfFunctions()->FindObject("palette");
      if (palette) {
        painter::adjustPalette(h, 0.92);
      }
    }

    ++idir;
    int nNaN = isNaNdX + isNaNdY + isNaNdZ;
    int nNaNS = isNaNdXS + isNaNdYS + isNaNdZS;
    const auto sNaN = fmt::format("NaN: {} - {} {} {} {}", inFile, nNaN, isNaNdX, isNaNdY, isNaNdZ);
    const auto sNaNS = fmt::format("NaNS: {} - {} {} {} {}", inFile, nNaNS, isNaNdXS, isNaNdYS, isNaNdZS);
    if (nNaN > 0) {
      LOGP(error, "{}", sNaN);
    } else {
      LOGP(info, "{}", sNaN);
    }
    if (nNaNS > 0) {
      LOGP(error, "{}", sNaNS);
    } else {
      LOGP(info, "{}", sNaNS);
    }
  }

  // ===| dX/dX vs row |========================================================
  //
  auto cdXZvsRow = new TCanvas(fmt::format("cdXZvsRow{}", outFileNameAdd).data(), "dX/dZ vs row", 1500, 900);
  cdXZvsRow->SetRightMargin(0.01);
  cdXZvsRow->SetBottomMargin(0.15);
  cdXZvsRow->Divide(1, 4, -1, -1);
  cdXZvsRow->cd(1);
  gPad->SetGrid();
  hsdXvsRowA->Draw("nostack");
  setSizes(hsdXvsRowA->GetYaxis(), 0.1, 0.4, 0.08);
  setSizes(hsdXvsRowA->GetXaxis(), 0.1, 0.4, 0.08);
  auto leg = gPad->BuildLegend(0.5, 0.5, 0.9, 0.9);
  leg->SetMargin(0.05);
  cdXZvsRow->cd(2);
  gPad->SetGrid();
  hsdXvsRowC->Draw("nostack");
  setSizes(hsdXvsRowC->GetYaxis(), 0.1, 0.4, 0.08);
  setSizes(hsdXvsRowC->GetXaxis(), 0.1, 0.4, 0.08);
  cdXZvsRow->cd(3);
  gPad->SetGrid();
  hsdZvsRowA->Draw("nostack");
  setSizes(hsdZvsRowA->GetYaxis(), 0.1, 0.4, 0.08);
  setSizes(hsdZvsRowA->GetXaxis(), 0.1, 0.4, 0.08);
  cdXZvsRow->cd(4);
  gPad->SetGrid();
  hsdZvsRowC->Draw("nostack");
  setSizes(hsdZvsRowC->GetYaxis(), 0.1, 0.4, 0.08);
  setSizes(hsdZvsRowC->GetXaxis(), 0.1, 0.4, 0.08);

  arrCanvases1D.Add(cdXZvsRow);

  // ===| save canvases |=======================================================
  //
  o2::tpc::utils::saveCanvases(arrCanvases1D, "./", "png,png", fmt::format("voxResQA{}_1D.root", outFileNameAdd.data()));
  o2::tpc::utils::saveCanvases(arrCanvases2D, "./", "png,png", fmt::format("voxResQA{}_2D.root", outFileNameAdd.data()));
}

std::string getString(TList* l, const std::string& name, std::string defaultVal)
{
  if (!l || !l->FindObject(name.data())) {
    return defaultVal;
  }
  return l->FindObject(name.data())->GetTitle();
}

int getNbins(const std::string& name, int defaultVal, const char delim)
{
  if (name.find(delim) != name.npos) {
    return std::count(name.begin(), name.end(), delim) + 1;
  }
  return std::stoi(name);
}

void setStyle(TH1* hist, int idir)
{
  const std::vector<Color_t> colors = {kRed + 2, kOrange + 1, kGreen + 2, kAzure + 10, kBlue + 2, kMagenta + 1};
  const std::vector<int> markers{20, 24, 21, 25, 47, 46, 34, 28};
  const std::vector<ELineStyle> styles{kSolid, kDashed, kDotted, kDashDotted};

  hist->SetMarkerSize(1);
  hist->SetMarkerColor(colors[idir % colors.size()]);
  hist->SetLineColor(colors[idir % colors.size()]);
  hist->SetMarkerStyle(markers[idir % markers.size()]);
  hist->SetLineStyle(styles[(idir / colors.size()) % styles.size()]);
}

void setSizes(TAxis* axis, float titleSize, float titleOffset, float labelSize)
{
  axis->SetTitleSize(titleSize);
  axis->SetTitleOffset(titleOffset);
  axis->SetLabelSize(labelSize);
}
