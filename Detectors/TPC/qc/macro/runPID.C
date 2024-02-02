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
#include <cstdint>
#include <iterator>
#include <string_view>
#include <vector>
#include <fmt/format.h>

#include "TTree.h"
#include "TFile.h"
#include "TStyle.h"
#include "TCanvas.h"
#include <TH1.h>
#include <TH2.h>
#include "DataFormatsTPC/TrackTPC.h"
#include "DataFormatsTPC/TrackCuts.h"
#include "TPCBase/CalDet.h"
#include "TPCBase/Painter.h"
#include "TPCBase/Utils.h"
#include "DataFormatsTPC/ClusterNative.h"
#include "TPCQC/PID.h"
#include "TPCQC/Helpers.h"
#endif

using namespace o2::tpc;

// Cuts
const int mipTot = 50;
const int mipMax = 50;

TCanvas* draw(std::vector<std::unique_ptr<TH1>>& histdEdxTot, std::vector<std::unique_ptr<TH1>>& histdEdxMax, std::string_view opt /*= ""*/, std::string_view add /*= ""*/, bool logx /*= false*/, bool logy /*= false*/, bool logz /*= false*/, int mipTot, int mipMax)
{
  auto* cdEdx = new TCanvas(fmt::format("cdEdx{}", add).data(), fmt::format("dEdx {}", add).data(), 1500, 500);
  if (histdEdxTot.size() == 1) {
    cdEdx->Divide(2, histdEdxTot.size());
  } else {
    cdEdx->Divide(histdEdxTot.size(), 2);
  }

  for (size_t idEdxType = 0; idEdxType < histdEdxTot.size(); ++idEdxType) {
    cdEdx->cd(idEdxType + 1);
    gPad->SetLogx(logx);
    gPad->SetLogy(logy);
    gPad->SetLogz(logz);
    auto& hTot = histdEdxTot[idEdxType];
    hTot->Draw(opt.data());

    if (histdEdxTot.size() == 1) {
      cdEdx->cd(idEdxType + 1 + 1);
    } else {
      cdEdx->cd(idEdxType + 1 + 5);
    }
    gPad->SetLogx(logx);
    gPad->SetLogy(logy);
    gPad->SetLogz(logz);
    auto& hMax = histdEdxMax[idEdxType];
    hMax->Draw(opt.data());
  }
  return cdEdx;
}

void runPID(std::string outputFileName = "PID", std::string_view inputFileName = "tpctracks.root", const size_t maxTracks = 0)
{
  // ===| track file and tree |=================================================
  auto file = TFile::Open(inputFileName.data());
  auto tree = (TTree*)file->Get("tpcrec");
  if (tree == nullptr) {
    std::cout << "Error getting tree\n";
    return;
  }

  // ===| branch setup |==========================================================
  std::vector<TrackTPC>* tpcTracks = nullptr;
  tree->SetBranchAddress("TPCTracks", &tpcTracks);

  // ===| create PID object |=====================================================
  qc::PID pid;
  // set elementart cuts for PID.  nClusterCut = 60, AbsTgl  = 1.,MindEdxTot = 10.0,MaxdEdxTot = 70., MinpTPC = 0.05, MaxpTPC = 20., MinpTPCMIPs = 0.45, MaxpTPCMIPs = 0.55
  pid.setPIDCuts(60, 1., 10.0, 70., 0.05, 20., 0.45, 0.55);
  pid.initializeHistograms();
  gStyle->SetPalette(kCividis);
  qc::helpers::setStyleHistogramsInMap(pid.getMapOfHisto());

  // ===| event loop |============================================================
  for (int i = 0; i < tree->GetEntriesFast(); ++i) {
    tree->GetEntry(i);
    size_t nTracks = (maxTracks > 0) ? std::min(tpcTracks->size(), maxTracks) : tpcTracks->size();
    // ---| track loop |---
    for (int k = 0; k < nTracks; k++) {
      auto track = (*tpcTracks)[k];
      pid.processTrack(track, nTracks);
    }
  }

  // ===| create canvas |========================================================
  std::unordered_map<std::string_view, std::vector<std::unique_ptr<TH1>>>& mMapOfHisto = pid.getMapOfHisto();

  std::vector<std::unique_ptr<TH1>>& histdEdxTotVspPos = (std::vector<std::unique_ptr<TH1>>&)mMapOfHisto["hdEdxTotVspPos"];
  std::vector<std::unique_ptr<TH1>>& histdEdxTotVspNeg = (std::vector<std::unique_ptr<TH1>>&)mMapOfHisto["hdEdxTotVspNeg"];
  std::vector<std::unique_ptr<TH1>>& histNClsPID = (std::vector<std::unique_ptr<TH1>>&)mMapOfHisto["hNClsPID"];
  std::vector<std::unique_ptr<TH1>>& histNClsSubPID = (std::vector<std::unique_ptr<TH1>>&)mMapOfHisto["hNClsSubPID"];
  std::vector<std::unique_ptr<TH1>>& histdEdxVsPhi = (std::vector<std::unique_ptr<TH1>>&)mMapOfHisto["hdEdxVsPhi"];
  std::vector<std::unique_ptr<TH1>>& histdEdxVsTgl = (std::vector<std::unique_ptr<TH1>>&)mMapOfHisto["hdEdxVsTgl"];
  std::vector<std::unique_ptr<TH1>>& histdEdxVsncls = (std::vector<std::unique_ptr<TH1>>&)mMapOfHisto["hdEdxVsncls"];
  std::vector<std::unique_ptr<TH1>>& histdEdxTotVspBeforeCuts = (std::vector<std::unique_ptr<TH1>>&)mMapOfHisto["hdEdxTotVspBeforeCuts"];
  std::vector<std::unique_ptr<TH1>>& histdEdxMaxVspBeforeCuts = (std::vector<std::unique_ptr<TH1>>&)mMapOfHisto["hdEdxMaxVspBeforeCuts"];
  std::vector<std::unique_ptr<TH1>>& histdEdxVsPhiMipsAside = (std::vector<std::unique_ptr<TH1>>&)mMapOfHisto["hdEdxVsPhiMipsAside"];
  std::vector<std::unique_ptr<TH1>>& histdEdxVsPhiMipsCside = (std::vector<std::unique_ptr<TH1>>&)mMapOfHisto["hdEdxVsPhiMipsCside"];

  std::vector<std::unique_ptr<TH1>>& histMIPNclVsTgl = (std::vector<std::unique_ptr<TH1>>&)mMapOfHisto["hMIPNclVsTgl"];
  std::vector<std::unique_ptr<TH1>>& histMIPNclVsTglSub = (std::vector<std::unique_ptr<TH1>>&)mMapOfHisto["hMIPNclVsTglSub"];

  std::vector<std::unique_ptr<TH1>>& histdEdxTot = (std::vector<std::unique_ptr<TH1>>&)mMapOfHisto["hdEdxTotVsp"];
  std::vector<std::unique_ptr<TH1>>& histdEdxMax = (std::vector<std::unique_ptr<TH1>>&)mMapOfHisto["hdEdxMaxVsp"];

  std::vector<std::unique_ptr<TH1>>& histdEdxTotMIP = (std::vector<std::unique_ptr<TH1>>&)mMapOfHisto["hdEdxTotMIP"];
  std::vector<std::unique_ptr<TH1>>& histdEdxMaxMIP = (std::vector<std::unique_ptr<TH1>>&)mMapOfHisto["hdEdxMaxMIP"];

  std::vector<std::unique_ptr<TH1>>& histdEdxTotMIPTgl = (std::vector<std::unique_ptr<TH1>>&)mMapOfHisto["hdEdxTotMIPVsTgl"];
  std::vector<std::unique_ptr<TH1>>& histdEdxMaxMIPTgl = (std::vector<std::unique_ptr<TH1>>&)mMapOfHisto["hdEdxMaxMIPVsTgl"];

  std::vector<std::unique_ptr<TH1>>& histdEdxTotMIPSnp = (std::vector<std::unique_ptr<TH1>>&)mMapOfHisto["hdEdxTotMIPVsSnp"];
  std::vector<std::unique_ptr<TH1>>& histdEdxMaxMIPSnp = (std::vector<std::unique_ptr<TH1>>&)mMapOfHisto["hdEdxMaxMIPVsSnp"];

  std::vector<std::unique_ptr<TH1>>& histdEdxTotMIPNcl = (std::vector<std::unique_ptr<TH1>>&)mMapOfHisto["hdEdxTotMIPVsNcl"];
  std::vector<std::unique_ptr<TH1>>& histdEdxMaxMIPNcl = (std::vector<std::unique_ptr<TH1>>&)mMapOfHisto["hdEdxMaxMIPVsNcl"];

  std::vector<std::unique_ptr<TH1>>& histdEdxTotMIPSec = (std::vector<std::unique_ptr<TH1>>&)mMapOfHisto["hdEdxTotMIPVsSec"];
  std::vector<std::unique_ptr<TH1>>& histdEdxMaxMIPSec = (std::vector<std::unique_ptr<TH1>>&)mMapOfHisto["hdEdxMaxMIPVsSec"];

  // dEdx vs. p
  auto cdEdxP = draw(histdEdxTot, histdEdxMax, "colz", "P", true, true, true, mipTot, mipMax);

  // dEdx MIP
  auto* cdEdxMIP = draw(histdEdxTotMIP, histdEdxMaxMIP, "", "MIP", false, false, true, mipTot, mipMax); // new TCanvas("cdEdxMIP", "dEdx MIP", 1500, 500);

  // dEdx MIP vs Tgl
  auto* cdEdxMIPTgl = draw(histdEdxTotMIPTgl, histdEdxMaxMIPTgl, "colz", "MIP_Tgl", false, false, true, mipTot, mipMax); // new TCanvas("cdEdxMIPTgl", "dEdx MIPTgl", 1500, 500);

  // dEdx MIP vs Snp
  auto* cdEdxMIPSnp = draw(histdEdxTotMIPSnp, histdEdxMaxMIPSnp, "colz", "MIP_Snp", false, false, true, mipTot, mipMax); // new TCanvas("cdEdxMIPSnp", "dEdx MIPSnp", 1500, 500);

  // dEdx MIP vs Tgl
  auto* cdEdxMIPNcl = draw(histdEdxTotMIPNcl, histdEdxMaxMIPNcl, "colz", "MIP_Ncl", false, false, true, mipTot, mipMax); // new TCanvas("cdEdxMIPNcl", "dEdx MIPNcl", 1500, 500);

  // dEdx MIP vs sector
  auto* cdEdxMIPSec = draw(histdEdxTotMIPSec, histdEdxMaxMIPSec, "colz", "MIP_Sec", false, false, true, mipTot, mipMax); // new TCanvas("cdEdxMIPSec", "dEdx MIPSec", 1500, 500);

  //--------------------------------------------
  // dEdx vs. p  sign
  auto cdEdxPTotSign = draw(histdEdxTotVspPos, histdEdxTotVspNeg, "colz", "Sign", true, true, true, mipTot, mipMax);

  // NClusters
  auto cNClsPID = draw(histNClsPID, histNClsSubPID, " ", "Ncluster", true, true, true, mipTot, mipMax);

  // dEdx vs phi and tgl
  auto cdEdxPhiandTgl = draw(histdEdxVsPhi, histdEdxVsTgl, "colz", "PhiAndTgl", true, true, true, mipTot, mipMax);

  // dEdx vs n cluseter and tgl
  auto cdEdxNclusterandTgl = draw(histdEdxVsncls, histdEdxVsTgl, "colz", "NclusterAndTgl", true, true, true, mipTot, mipMax);

  // dEdx MIP and TOT vs p before cuts
  auto cdEdxPMIPandTOT = draw(histdEdxTotVspBeforeCuts, histdEdxMaxVspBeforeCuts, "colz", "MIPAndTotBeforeCuts", true, true, true, mipTot, mipMax);

  // dEdx vs phi A and C side
  auto cdEdxPhiSides = draw(histdEdxVsPhiMipsAside, histdEdxVsPhiMipsCside, "colz", "phi_A_and_C_side", true, true, true, mipTot, mipMax);

  // N cluster MIP vs tgl
  auto cNclustervsTglMIPs = draw(histMIPNclVsTgl, histMIPNclVsTglSub, "colz", "MIP_ncls", true, true, true, mipTot, mipMax);

  std::vector<TCanvas*> canvases;
  canvases.emplace_back(cdEdxP);
  canvases.emplace_back(cdEdxMIP);
  canvases.emplace_back(cdEdxMIPTgl);
  canvases.emplace_back(cdEdxMIPSnp);
  canvases.emplace_back(cdEdxMIPNcl);
  canvases.emplace_back(cdEdxMIPSec);

  canvases.emplace_back(cdEdxPTotSign);
  canvases.emplace_back(cNClsPID);
  canvases.emplace_back(cdEdxPhiandTgl);
  canvases.emplace_back(cdEdxNclusterandTgl);
  canvases.emplace_back(cdEdxPMIPandTOT);
  canvases.emplace_back(cdEdxPhiSides);
  canvases.emplace_back(cNclustervsTglMIPs);

  if (outputFileName.find(".root") != std::string::npos) {
    outputFileName.resize(outputFileName.size() - 5);
  }

  //===| dump canvases to a file |=============================================
  std::string canvasFile = outputFileName + "_canvas.root";
  utils::saveCanvases(canvases, "./", "", canvasFile);

  //===| dump histograms to a file |=============================================
  std::string histFile = outputFileName + ".root";
  pid.dumpToFile(histFile);

  pid.resetHistograms();
}
