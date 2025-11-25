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

/// \file PlotDCA.C
/// \brief Simple macro to plot ITS3 impact parameter resolution

#if !defined(__CLING__) || defined(__ROOTCLING__)
#include <memory>

#include <TROOT.h>
#include <TFile.h>
#include <TF1.h>
#include <TH1F.h>
#include <TH2F.h>
#include <TTree.h>

#include "ReconstructionDataFormats/GlobalTrackID.h"
#include "ReconstructionDataFormats/Track.h"
#include "ReconstructionDataFormats/DCA.h"
#include "SimulationDataFormat/MCTrack.h"
#endif

using GTrackID = o2::dataformats::GlobalTrackID;

static std::string SanitizeSourceName(std::string_view raw)
{
  std::string s(raw);
  s.erase(std::remove(s.begin(), s.end(), '-'), s.end());
  return s;
}

void PlotDCA(const char* fName = "its3TrackStudy.root")
{
  TH1::SetDefaultSumw2();
  std::unique_ptr<TFile> inFile(TFile::Open(fName));
  auto tree = inFile->Get<TTree>("dca");

  int src; // track type
  tree->SetBranchAddress("src", &src);
  o2::dataformats::DCA* dca{nullptr};
  tree->SetBranchAddress("dca2MC", &dca);
  o2::track::TrackParCov* trk{nullptr};
  tree->SetBranchAddress("trk", &trk);
  o2::MCTrack* mcTrk{nullptr};
  tree->SetBranchAddress("mcTrk", &mcTrk);

  const int nPtBins = 35;
  const double ptLimits[nPtBins] = {0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7, 0.8, 0.9, 1., 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2., 2.2, 2.5, 3., 4., 5., 6., 8., 10., 15., 20.};
  const int yDCABins{1000};
  const float yDCARange{500};
  const int nSpecies = 5;
  std::array<int, nSpecies> pdgCodes{-1, 11, 211, 321, 2212};
  auto fGaus = new TF1("fGaus", "gaus", -200., 200.);
  std::map<int, std::string> partNames = {
    {-1, "All"},
    {11, "Electrons"},
    {211, "Pions"},
    {321, "Kaons"},
    {2212, "Protons"}};

  std::map<int, std::map<int, TH2F*>> hMapDCAxyVsPtAllLayers;    // species -> [src, {dca}]
  std::map<int, std::map<int, TH1F*>> hMapResDCAxyVsPtAllLayers; // species -> [src, {dca}]
  std::map<int, std::map<int, TH2F*>> hMapDCAzVsPtAllLayers;     // species -> [src, {dca}]
  std::map<int, std::map<int, TH1F*>> hMapResDCAzVsPtAllLayers;  // species -> [src, {dca}]
  std::map<int, std::map<int, TH2F*>> hMapDeltaPtVsPtAllLayers;  // species -> [src, {dca}]
  std::map<int, std::map<int, TH1F*>> hMapResPtVsPtAllLayers;    // species -> [src, {dca}]
  for (const auto& [sPDG, sName] : partNames) {
    std::map<int, TH2F*> histsDCAxy, histsDCAz, histsDeltaPt;
    std::map<int, TH1F*> histsResDCAxy, histsResDCAz, histsResDeltaPt;

    for (int cis = 0; cis < GTrackID::NSources; ++cis) {
      const auto cdm = GTrackID::getSourceDetectorsMask(cis);
      if (!cdm[GTrackID::ITS]) {
        continue; // keep same logic as original
      }

      const std::string srcRaw = GTrackID::getSourceName(cis);
      const std::string src = SanitizeSourceName(srcRaw);

      histsDCAxy[cis] = new TH2F(Form("hDCAxyVsPtAllLayers_%s_%s", sName.c_str(), src.c_str()), Form("%s;#it{p}_{T,MC} (GeV/#it{c});DCA_{#it{xy}} (#mum);entries", srcRaw.c_str()), nPtBins - 1, ptLimits, yDCABins, -yDCARange, yDCARange);

      histsResDCAxy[cis] = new TH1F(Form("hResDCAxyVsPtAllLayers_%s_%s", sName.c_str(), src.c_str()), Form("%s;#it{p}_{T,MC} (GeV/#it{c});#sigma(DCA_{#it{xy}}) (#mum);entries", srcRaw.c_str()), nPtBins - 1, ptLimits);

      histsDCAz[cis] = new TH2F(Form("hDCAzVsPtAllLayers_%s_%s", sName.c_str(), src.c_str()), Form("%s;#it{p}_{T,MC} (GeV/#it{c});DCA_{#it{z}} (#mum);entries", srcRaw.c_str()), nPtBins - 1, ptLimits, yDCABins, -yDCARange, yDCARange);

      histsResDCAz[cis] = new TH1F(Form("hResDCAzVsPtAllLayers_%s_%s", sName.c_str(), src.c_str()), Form("%s;#it{p}_{T,MC} (GeV/#it{c});#sigma(DCA_{#it{z}}) (#mum);entries", srcRaw.c_str()), nPtBins - 1, ptLimits);

      histsDeltaPt[cis] = new TH2F(Form("hDeltaPtVsPtAllLayers_%s_%s", sName.c_str(), src.c_str()), Form("%s;#it{p}_{T,MC} (GeV/#it{c});#Delta_{#it{p}_{T}}/#it{p}_{T};entries", srcRaw.c_str()), nPtBins - 1, ptLimits, 200, -0.2, 0.2);

      histsResDeltaPt[cis] = new TH1F(Form("hResDeltaPtVsPtAllLayers_%s_%s", sName.c_str(), src.c_str()), Form("%s;#it{p}_{T,MC} (GeV/#it{c});#sigma(#Delta#it{p}_{T}/#it{p}_{T});entries", srcRaw.c_str()), nPtBins - 1, ptLimits);
    }

    hMapDCAxyVsPtAllLayers[sPDG] = std::move(histsDCAxy);
    hMapResDCAxyVsPtAllLayers[sPDG] = std::move(histsResDCAxy);
    hMapDCAzVsPtAllLayers[sPDG] = std::move(histsDCAz);
    hMapResDCAzVsPtAllLayers[sPDG] = std::move(histsResDCAz);
    hMapDeltaPtVsPtAllLayers[sPDG] = std::move(histsDeltaPt);
    hMapResPtVsPtAllLayers[sPDG] = std::move(histsResDeltaPt);
  }

  for (int iEntry = 0; tree->LoadTree(iEntry) >= 0; ++iEntry) {
    tree->GetEntry(iEntry);
    if (!mcTrk->isPrimary()) {
      continue;
    }
    auto pdg = std::abs(mcTrk->GetPdgCode());
    if (pdg != 11 && pdg != 211 && pdg != 321 && pdg != 2212) {
      continue;
    }
    auto ptReco = trk->getPt();
    auto ptGen = mcTrk->GetPt();
    auto deltaPt = (1. / ptReco - 1. / ptGen) / (1. / ptGen);
    auto dcaXY = dca->getY() * 10000.;
    auto dcaZ = dca->getZ() * 10000.;
    auto phiReco = trk->getPhi();

    for (int spe : {-1, pdg}) {
      hMapDeltaPtVsPtAllLayers[spe][src]->Fill(ptGen, deltaPt);
      hMapDCAxyVsPtAllLayers[spe][src]->Fill(ptGen, dcaXY);
      hMapDCAzVsPtAllLayers[spe][src]->Fill(ptGen, dcaZ);
    }
  }

  const char* fitOpt{"QWMER"};
  for (const auto& [sPDG, sName] : partNames) {
    for (int cis = 0; cis < GTrackID::NSources; cis++) {
      const auto cdm = GTrackID::getSourceDetectorsMask(cis);
      if (!cdm[GTrackID::ITS]) {
        continue;
      }
      for (auto iPt{0}; iPt < nPtBins; ++iPt) {
        auto ptMin = hMapDCAxyVsPtAllLayers[sPDG][cis]->GetXaxis()->GetBinLowEdge(iPt + 1);
        float minFit = (ptMin < 1.) ? -200. : -50.;
        float maxFit = (ptMin < 1.) ? 200. : 50.;
        auto doProjection = [&](auto& hIn, auto& hOut, bool useRange = true) {
          auto hProj = hIn[sPDG][cis]->ProjectionY(Form("%s_%d", hOut[sPDG][cis]->GetName(), iPt), iPt + 1, iPt + 1);
          if (hProj->GetEntries() < 100) {
            return;
          }
          if (useRange) {
            hProj->Fit("fGaus", fitOpt, "", minFit, maxFit);
          } else {
            hProj->Fit("fGaus", fitOpt);
          }
          hOut[sPDG][cis]->SetBinContent(iPt + 1, fGaus->GetParameter(2));
          hOut[sPDG][cis]->SetBinError(iPt + 1, fGaus->GetParError(2));
        };

        doProjection(hMapDeltaPtVsPtAllLayers, hMapResPtVsPtAllLayers, false);
        doProjection(hMapDCAxyVsPtAllLayers, hMapResDCAxyVsPtAllLayers);
        doProjection(hMapDCAzVsPtAllLayers, hMapResDCAzVsPtAllLayers);
      }
    }
  }

  TFile outFile("plotDCA.root", "RECREATE");
  for (const auto& [sPDG, sName] : partNames) {
    outFile.mkdir(sName.c_str());
    outFile.cd(sName.c_str());
    for (int cis = 0; cis < GTrackID::NSources; cis++) {
      const auto cdm = GTrackID::getSourceDetectorsMask(cis);
      if (!cdm[GTrackID::ITS]) {
        continue;
      }
      const std::string srcRaw = GTrackID::getSourceName(cis);
      const std::string src = SanitizeSourceName(srcRaw);
      gDirectory->mkdir(src.c_str());
      gDirectory->cd(src.c_str());

      hMapDCAxyVsPtAllLayers[sPDG][cis]->Write();
      hMapResDCAxyVsPtAllLayers[sPDG][cis]->Write();
      hMapDCAzVsPtAllLayers[sPDG][cis]->Write();
      hMapResDCAzVsPtAllLayers[sPDG][cis]->Write();
      hMapDeltaPtVsPtAllLayers[sPDG][cis]->Write();
      hMapResPtVsPtAllLayers[sPDG][cis]->Write();

      outFile.cd(sName.c_str());
    }

    outFile.cd();
  }
}
