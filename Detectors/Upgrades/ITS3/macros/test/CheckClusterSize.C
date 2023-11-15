// Copyright 2020-2022 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file CheckClusterSize.C
/// \brief analyze ITS3 cluster sizes
/// \dependencies CreateDictionariesITS3.C
/// \author felix.schlepper@cern.ch

#if !defined(__CLING__) || defined(__ROOTCLING__)
#include <TCanvas.h>
#include <TFile.h>
#include <TGraphErrors.h>
#include <TH2F.h>
#include <TLegend.h>
#include <TMultiGraph.h>
#include <TROOT.h>
#include <TStyle.h>
#include <TTree.h>
#include <TStopwatch.h>
#include <TPDGCode.h>

#include <array>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <tuple>
#include <vector>

#define ENABLE_UPGRADES
#include "DataFormatsITS3/CompCluster.h"
#include "DataFormatsITSMFT/ROFRecord.h"
#include "DetectorsCommonDataFormats/DetID.h"
#include "DetectorsCommonDataFormats/DetectorNameConf.h"
#include "ITS3Reconstruction/TopologyDictionary.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/MCEventHeader.h"
#include "SimulationDataFormat/MCTrack.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#endif

static constexpr int nLayers = 4; // 3 Layers + 1 combined outer layer

struct ParticleInfo {
  int event{};
  int pdg{};
  double pt{};
  double eta{};
  double phi{};
  bool isPrimary{false};
};

using o2::its3::CompClusterExt;
using ROFRec = o2::itsmft::ROFRecord;

void checkFile(const std::unique_ptr<TFile>& file);

inline auto hist_map(unsigned short id)
{
  return std::clamp(id, static_cast<unsigned short>(0), static_cast<unsigned short>(6)) / 2;
}

void CheckClusterSize(std::string clusFileName = "o2clus_it3.root",
                      std::string kineFileName = "o2sim_Kine.root",
                      std::string dictFileName = "", bool batch = true)
{
  gROOT->SetBatch(batch);
  TStopwatch sw;
  sw.Start();
  // TopologyDictionary
  if (dictFileName.empty()) {
    dictFileName =
      o2::base::DetectorNameConf::getAlpideClusterDictionaryFileName(
        o2::detectors::DetID::IT3, "", "root");
  }
  o2::its3::TopologyDictionary dict;
  std::ifstream file(dictFileName.c_str());
  if (file.good()) {
    LOG(info) << "Running with dictionary: " << dictFileName.c_str();
    dict.readFromFile(dictFileName);
  } else {
    LOG(info) << "Running without dictionary !";
  }

  // Histograms
  constexpr int maxClusterSize = 50;
  TH1F hOuterBarrel("outerbarrel", "ClusterSize in OuterBarrel", maxClusterSize, 0, maxClusterSize);
  std::vector<TH1I> hPrimary;
  std::vector<TH2D> hPrimaryEta;
  std::vector<TH2D> hPrimaryPt;
  std::vector<TH2D> hPrimaryPhi;
  std::vector<TH1I> hSecondary;
  std::vector<TH2D> hSecondaryEta;
  std::vector<TH2D> hSecondaryPt;
  std::vector<TH2D> hSecondaryPhi;
  std::vector<TH1I> hProtonPrimary;
  std::vector<TH2D> hProtonPrimaryEta;
  std::vector<TH2D> hProtonPrimaryPt;
  std::vector<TH2D> hProtonPrimaryPhi;
  std::vector<TH1I> hProtonSecondary;
  std::vector<TH2D> hProtonSecondaryEta;
  std::vector<TH2D> hProtonSecondaryPt;
  std::vector<TH2D> hProtonSecondaryPhi;
  std::vector<TH1I> hPionPrimary;
  std::vector<TH2D> hPionPrimaryEta;
  std::vector<TH2D> hPionPrimaryPt;
  std::vector<TH2D> hPionPrimaryPhi;
  std::vector<TH1I> hPionSecondary;
  std::vector<TH2D> hPionSecondaryEta;
  std::vector<TH2D> hPionSecondaryPt;
  std::vector<TH2D> hPionSecondaryPhi;
  std::vector<TH1I> hKaonPrimary;
  std::vector<TH2D> hKaonPrimaryEta;
  std::vector<TH2D> hKaonPrimaryPt;
  std::vector<TH2D> hKaonPrimaryPhi;
  std::vector<TH1I> hKaonSecondary;
  std::vector<TH2D> hKaonSecondaryEta;
  std::vector<TH2D> hKaonSecondaryPt;
  std::vector<TH2D> hKaonSecondaryPhi;
  std::vector<TH1I> hOtherPrimary;
  std::vector<TH2D> hOtherPrimaryEta;
  std::vector<TH2D> hOtherPrimaryPt;
  std::vector<TH2D> hOtherPrimaryPhi;
  std::vector<TH1I> hOtherSecondary;
  std::vector<TH2D> hOtherSecondaryEta;
  std::vector<TH2D> hOtherSecondaryPt;
  std::vector<TH2D> hOtherSecondaryPhi;
  for (int i = 0; i < 4; ++i) {
    hPrimary.emplace_back(Form("primary/L%d", i), Form("L%d Primary Cluster Size", i), maxClusterSize, 0, maxClusterSize);
    hPrimaryEta.emplace_back(Form("primary/EtaL%d", i), Form("L%d Primary Cluster Size vs Eta", i), maxClusterSize, 0, maxClusterSize, 100, -3.0, 3.0);
    hPrimaryPt.emplace_back(Form("primary/Pt%d", i), Form("L%d Primary Cluster Size vs Pt", i), maxClusterSize, 0, maxClusterSize, 100, 0.0, 10.0);
    hPrimaryPhi.emplace_back(Form("primary/Phi%d", i), Form("L%d Primary Cluster Size vs Phi", i), maxClusterSize, 0, maxClusterSize, 100, 0., 2 * o2::constants::math::PI);
    hSecondary.emplace_back(Form("seconday/L%d", i), Form("L%d Secondary Cluster Size", i), maxClusterSize, 0, maxClusterSize);
    hSecondaryEta.emplace_back(Form("seconday/EtaL%d", i), Form("L%d Secondary Cluster Size vs Eta", i), maxClusterSize, 0, maxClusterSize, 100, -3.0, 3.0);
    hSecondaryPt.emplace_back(Form("seconday/Pt%d", i), Form("L%d Secondary Cluster Size vs Pt", i), maxClusterSize, 0, maxClusterSize, 100, 0.0, 10.0);
    hSecondaryPhi.emplace_back(Form("seconday/Phi%d", i), Form("L%d Secondary Cluster Size vs Phi", i), maxClusterSize, 0, maxClusterSize, 100, 0., 2 * o2::constants::math::PI);

    hProtonPrimary.emplace_back(Form("proton/primary/L%d", i), Form("Proton - L%d Primary Cluster Size", i), maxClusterSize, 0, maxClusterSize);
    hProtonPrimaryEta.emplace_back(Form("proton/primary/EtaL%d", i), Form("Proton - L%d Primary Cluster Size vs Eta", i), maxClusterSize, 0, maxClusterSize, 100, -3.0, 3.0);
    hProtonPrimaryPt.emplace_back(Form("proton/primary/Pt%d", i), Form("Proton - L%d Primary Cluster Size vs Pt", i), maxClusterSize, 0, maxClusterSize, 100, 0.0, 10.0);
    hProtonPrimaryPhi.emplace_back(Form("proton/primary/Phi%d", i), Form("Proton - L%d Primary Cluster Size vs Phi", i), maxClusterSize, 0, maxClusterSize, 100, 0., 2 * o2::constants::math::PI);
    hProtonSecondary.emplace_back(Form("proton/seconday/L%d", i), Form("Proton - L%d Secondary Cluster Size", i), maxClusterSize, 0, maxClusterSize);
    hProtonSecondaryEta.emplace_back(Form("proton/seconday/EtaL%d", i), Form("Proton - L%d Secondary Cluster Size vs Eta", i), maxClusterSize, 0, maxClusterSize, 100, -3.0, 3.0);
    hProtonSecondaryPt.emplace_back(Form("proton/seconday/Pt%d", i), Form("Proton - L%d Secondary Cluster Size vs Pt", i), maxClusterSize, 0, maxClusterSize, 100, 0.0, 10.0);
    hProtonSecondaryPhi.emplace_back(Form("proton/seconday/Phi%d", i), Form("Proton - L%d Secondary Cluster Size vs Phi", i), maxClusterSize, 0, maxClusterSize, 100, 0., 2 * o2::constants::math::PI);

    hPionPrimary.emplace_back(Form("pion/primary/L%d", i), Form("Pion- L%d Primary Cluster Size", i), maxClusterSize, 0, maxClusterSize);
    hPionPrimaryEta.emplace_back(Form("pion/primary/EtaL%d", i), Form("Pion- L%d Primary Cluster Size vs Eta", i), maxClusterSize, 0, maxClusterSize, 100, -3.0, 3.0);
    hPionPrimaryPt.emplace_back(Form("pion/primary/Pt%d", i), Form("Pion- L%d Primary Cluster Size vs Pt", i), maxClusterSize, 0, maxClusterSize, 100, 0.0, 10.0);
    hPionPrimaryPhi.emplace_back(Form("pion/primary/Phi%d", i), Form("Pion- L%d Primary Cluster Size vs Phi", i), maxClusterSize, 0, maxClusterSize, 100, 0., 2 * o2::constants::math::PI);
    hPionSecondary.emplace_back(Form("pion/seconday/L%d", i), Form("Pion- L%d Secondary Cluster Size", i), maxClusterSize, 0, maxClusterSize);
    hPionSecondaryEta.emplace_back(Form("pion/seconday/EtaL%d", i), Form("Pion- L%d Secondary Cluster Size vs Eta", i), maxClusterSize, 0, maxClusterSize, 100, -3.0, 3.0);
    hPionSecondaryPt.emplace_back(Form("pion/seconday/Pt%d", i), Form("Pion- L%d Secondary Cluster Size vs Pt", i), maxClusterSize, 0, maxClusterSize, 100, 0.0, 10.0);
    hPionSecondaryPhi.emplace_back(Form("pion/seconday/Phi%d", i), Form("Pion- L%d Secondary Cluster Size vs Phi", i), maxClusterSize, 0, maxClusterSize, 100, 0., 2 * o2::constants::math::PI);

    hKaonPrimary.emplace_back(Form("kaon/primary/L%d", i), Form("Kaon- L%d Primary Cluster Size", i), maxClusterSize, 0, maxClusterSize);
    hKaonPrimaryEta.emplace_back(Form("kaon/primary/EtaL%d", i), Form("Kaon- L%d Primary Cluster Size vs Eta", i), maxClusterSize, 0, maxClusterSize, 100, -3.0, 3.0);
    hKaonPrimaryPt.emplace_back(Form("kaon/primary/Pt%d", i), Form("Kaon- L%d Primary Cluster Size vs Pt", i), maxClusterSize, 0, maxClusterSize, 100, 0.0, 10.0);
    hKaonPrimaryPhi.emplace_back(Form("kaon/primary/Phi%d", i), Form("Kaon- L%d Primary Cluster Size vs Phi", i), maxClusterSize, 0, maxClusterSize, 100, 0., 2 * o2::constants::math::PI);
    hKaonSecondary.emplace_back(Form("kaon/seconday/L%d", i), Form("Kaon- L%d Secondary Cluster Size", i), maxClusterSize, 0, maxClusterSize);
    hKaonSecondaryEta.emplace_back(Form("kaon/seconday/EtaL%d", i), Form("Kaon- L%d Secondary Cluster Size vs Eta", i), maxClusterSize, 0, maxClusterSize, 100, -3.0, 3.0);
    hKaonSecondaryPt.emplace_back(Form("kaon/seconday/Pt%d", i), Form("Kaon- L%d Secondary Cluster Size vs Pt", i), maxClusterSize, 0, maxClusterSize, 100, 0.0, 10.0);
    hKaonSecondaryPhi.emplace_back(Form("kaon/seconday/Phi%d", i), Form("Kaon- L%d Secondary Cluster Size vs Phi", i), maxClusterSize, 0, maxClusterSize, 100, 0., 2 * o2::constants::math::PI);

    hOtherPrimary.emplace_back(Form("other/primary/L%d", i), Form("Other - L%d Primary Cluster Size", i), maxClusterSize, 0, maxClusterSize);
    hOtherPrimaryEta.emplace_back(Form("other/primary/EtaL%d", i), Form("Other - L%d Primary Cluster Size vs Eta", i), maxClusterSize, 0, maxClusterSize, 100, -3.0, 3.0);
    hOtherPrimaryPt.emplace_back(Form("other/primary/Pt%d", i), Form("Other - L%d Primary Cluster Size vs Pt", i), maxClusterSize, 0, maxClusterSize, 100, 0.0, 10.0);
    hOtherPrimaryPhi.emplace_back(Form("other/primary/Phi%d", i), Form("Other - L%d Primary Cluster Size vs Phi", i), maxClusterSize, 0, maxClusterSize, 100, 0., 2 * o2::constants::math::PI);
    hOtherSecondary.emplace_back(Form("other/seconday/L%d", i), Form("Other - L%d Secondary Cluster Size", i), maxClusterSize, 0, maxClusterSize);
    hOtherSecondaryEta.emplace_back(Form("other/seconday/EtaL%d", i), Form("Other - L%d Secondary Cluster Size vs Eta", i), maxClusterSize, 0, maxClusterSize, 100, -3.0, 3.0);
    hOtherSecondaryPt.emplace_back(Form("other/seconday/Pt%d", i), Form("Other - L%d Secondary Cluster Size vs Pt", i), maxClusterSize, 0, maxClusterSize, 100, 0.0, 10.0);
    hOtherSecondaryPhi.emplace_back(Form("other/seconday/Phi%d", i), Form("Other - L%d Secondary Cluster Size vs Phi", i), maxClusterSize, 0, maxClusterSize, 100, 0., 2 * o2::constants::math::PI);
  }

  // Clusters
  std::unique_ptr<TFile> clusFile(TFile::Open(clusFileName.data()));
  checkFile(clusFile);
  auto clusTree = clusFile->Get<TTree>("o2sim");
  std::vector<CompClusterExt> clusArr;
  std::vector<CompClusterExt>* clusArrP{&clusArr};
  clusTree->SetBranchAddress("IT3ClusterComp", &clusArrP);
  std::vector<unsigned char> patterns;
  std::vector<unsigned char>* patternsPtr{&patterns};
  clusTree->SetBranchAddress("IT3ClusterPatt", &patternsPtr);

  // MC tracks
  std::unique_ptr<TFile> kineFile(TFile::Open(kineFileName.data()));
  checkFile(kineFile);
  auto mcTree = kineFile->Get<TTree>("o2sim");
  mcTree->SetBranchStatus("*", false); // disable all branches
  mcTree->SetBranchStatus("MCTrack*", true);
  mcTree->SetBranchStatus("MCEventHeader*", true);

  std::vector<o2::MCTrack> mcArr;
  std::vector<o2::MCTrack>* mcArrP{&mcArr};
  mcTree->SetBranchAddress("MCTrack", &mcArrP);
  o2::dataformats::MCEventHeader* mcEvent = nullptr;
  mcTree->SetBranchAddress("MCEventHeader.", &mcEvent);

  // Cluster MC labels
  o2::dataformats::MCTruthContainer<o2::MCCompLabel>* clusLabArr = nullptr;
  clusTree->SetBranchAddress("IT3ClusterMCTruth", &clusLabArr);

  std::cout << "** Filling particle table ... " << std::flush;
  int lastEventIDcl = -1;
  auto nev = mcTree->GetEntriesFast();
  std::vector<std::vector<ParticleInfo>> info(nev);
  for (int iEntry = 0; mcTree->LoadTree(iEntry) >= 0; ++iEntry) { // loop over MC events
    mcTree->GetEvent(iEntry);
    info[iEntry].resize(mcArr.size());
    for (unsigned int mcI{0}; mcI < mcArr.size(); ++mcI) {
      const auto part = mcArr[mcI];
      info[iEntry][mcI].event = iEntry;
      info[iEntry][mcI].pdg = std::abs(part.GetPdgCode());
      info[iEntry][mcI].pt = part.GetPt();
      info[iEntry][mcI].phi = part.GetPhi();
      info[iEntry][mcI].eta = part.GetEta();
      if (std::sqrt(part.GetStartVertexCoordinatesX() * part.GetStartVertexCoordinatesX() + part.GetStartVertexCoordinatesY() * part.GetStartVertexCoordinatesY()) < 0.1) {
        info[iEntry][mcI].isPrimary = true;
      }
    }
  }
  std::cout << "  done." << std::endl;

  // ROFrecords
  std::vector<ROFRec> rofRecVec;
  std::vector<ROFRec>* rofRecVecP{&rofRecVec};
  clusTree->SetBranchAddress("IT3ClustersROF", &rofRecVecP);
  clusTree->GetEntry(0);
  int nROFRec = (int)rofRecVec.size();
  auto pattIt = patternsPtr->cbegin();

  for (int irof = 0; irof < nROFRec; irof++) {
    const auto& rofRec = rofRecVec[irof];
    // rofRec.print();

    for (int icl = 0; icl < rofRec.getNEntries(); icl++) {
      int clEntry = rofRec.getFirstEntry() + icl;
      const auto& cluster = clusArr[clEntry];
      // cluster.print();

      auto pattId = cluster.getPatternID();
      auto id = cluster.getSensorID();
      int clusterSize{-1};
      if (pattId == o2::its3::CompCluster::InvalidPatternID ||
          dict.isGroup(pattId)) {
        o2::itsmft::ClusterPattern patt(pattIt);
        clusterSize = patt.getNPixels();
      } else {
        clusterSize = dict.getNpixels(pattId);
      }

      const auto& label = (clusLabArr->getLabels(clEntry))[0];
      if (!label.isValid() || label.getSourceID() != 0 || !label.isCorrect()) {
        continue;
      }

      const int trackID = label.getTrackID();
      int evID = label.getEventID();
      const auto& pInfo = info[evID][trackID];
      if (id > 6) {
        hOuterBarrel.Fill(clusterSize);
      }

      if (pInfo.isPrimary) {
        hPrimary[hist_map(id)].Fill(clusterSize);
        hPrimaryEta[hist_map(id)].Fill(clusterSize, pInfo.eta);
        hPrimaryPt[hist_map(id)].Fill(clusterSize, pInfo.pt);
        hPrimaryPhi[hist_map(id)].Fill(clusterSize, pInfo.phi);
      } else {
        hSecondary[hist_map(id)].Fill(clusterSize);
        hSecondaryEta[hist_map(id)].Fill(clusterSize, pInfo.eta);
        hSecondaryPt[hist_map(id)].Fill(clusterSize, pInfo.pt);
        hSecondaryPhi[hist_map(id)].Fill(clusterSize, pInfo.phi);
      }
      if (pInfo.pdg == kProton) {
        if (pInfo.isPrimary) {
          hProtonPrimary[hist_map(id)].Fill(clusterSize);
          hProtonPrimaryEta[hist_map(id)].Fill(clusterSize, pInfo.eta);
          hProtonPrimaryPt[hist_map(id)].Fill(clusterSize, pInfo.pt);
          hProtonPrimaryPhi[hist_map(id)].Fill(clusterSize, pInfo.phi);
        } else {
          hProtonSecondary[hist_map(id)].Fill(clusterSize);
          hProtonSecondaryEta[hist_map(id)].Fill(clusterSize, pInfo.eta);
          hProtonSecondaryPt[hist_map(id)].Fill(clusterSize, pInfo.pt);
          hProtonSecondaryPhi[hist_map(id)].Fill(clusterSize, pInfo.phi);
        }
      } else if (pInfo.pdg == kPiPlus) {
        if (pInfo.isPrimary) {
          hProtonPrimary[hist_map(id)].Fill(clusterSize);
          hProtonPrimaryEta[hist_map(id)].Fill(clusterSize, pInfo.eta);
          hProtonPrimaryPt[hist_map(id)].Fill(clusterSize, pInfo.pt);
          hProtonPrimaryPhi[hist_map(id)].Fill(clusterSize, pInfo.phi);
        } else {
          hPionSecondary[hist_map(id)].Fill(clusterSize);
          hPionSecondaryEta[hist_map(id)].Fill(clusterSize, pInfo.eta);
          hPionSecondaryPt[hist_map(id)].Fill(clusterSize, pInfo.pt);
          hPionSecondaryPhi[hist_map(id)].Fill(clusterSize, pInfo.phi);
        }
      } else if (pInfo.pdg == kKPlus) {
        if (pInfo.isPrimary) {
          hKaonPrimary[hist_map(id)].Fill(clusterSize);
          hKaonPrimaryEta[hist_map(id)].Fill(clusterSize, pInfo.eta);
          hKaonPrimaryPt[hist_map(id)].Fill(clusterSize, pInfo.pt);
          hKaonPrimaryPhi[hist_map(id)].Fill(clusterSize, pInfo.phi);
        } else {
          hKaonSecondary[hist_map(id)].Fill(clusterSize);
          hKaonSecondaryEta[hist_map(id)].Fill(clusterSize, pInfo.eta);
          hKaonSecondaryPt[hist_map(id)].Fill(clusterSize, pInfo.pt);
          hKaonSecondaryPhi[hist_map(id)].Fill(clusterSize, pInfo.phi);
        }
      } else {
        if (pInfo.isPrimary) {
          hOtherPrimary[hist_map(id)].Fill(clusterSize);
          hOtherPrimaryEta[hist_map(id)].Fill(clusterSize, pInfo.eta);
          hOtherPrimaryPt[hist_map(id)].Fill(clusterSize, pInfo.pt);
          hOtherPrimaryPhi[hist_map(id)].Fill(clusterSize, pInfo.phi);
        } else {
          hOtherSecondary[hist_map(id)].Fill(clusterSize);
          hOtherSecondaryEta[hist_map(id)].Fill(clusterSize, pInfo.eta);
          hOtherSecondaryPt[hist_map(id)].Fill(clusterSize, pInfo.pt);
          hOtherSecondaryPhi[hist_map(id)].Fill(clusterSize, pInfo.phi);
        }
      }
    }
  }
  std::cout << "Done measuring cluster sizes:" << std::endl;
  for (int i = 0; i < nLayers; ++i) {
    std::cout << "* Layer " << i << ":\n";
    std::cout << "** Primary " << hPrimary[i].GetMean() << " +/- " << hPrimary[i].GetRMS() << "\n";
    std::cout << "** Secondary " << hSecondary[i].GetMean() << " +/- " << hSecondary[i].GetRMS() << std::endl;
  }
  std::unique_ptr<TFile> oFile(
    TFile::Open("checkClusterSize.root", "RECREATE"));
  checkFile(oFile);

  char const* name[nLayers] = {"L0", "L1", "L2", "OuterBarrel"};
  const double delta = 0.2;
  double x[nLayers], xP[nLayers], xS[nLayers], yP[nLayers], yS[nLayers], vyP[nLayers], vyS[nLayers];
  for (int i = 0; i < nLayers; ++i) {
    x[i] = i;
    xP[i] = i - delta;
    xS[i] = i + delta;
    yP[i] = hPrimary[i].GetMean();
    vyP[i] = hPrimary[i].GetRMS();
    yS[i] = hSecondary[i].GetMean();
    vyS[i] = hSecondary[i].GetRMS();
  }

  auto c1 = new TCanvas("c1", "A Simple Graph Example", 200, 10, 700, 500);
  auto h = new TH1F("h", "", nLayers, x[0] - 0.5, x[nLayers - 1] + 0.5);
  h->SetTitle("Cluster Sizes");
  h->GetYaxis()->SetTitleOffset(1.);
  h->GetXaxis()->SetTitleOffset(1.);
  h->GetYaxis()->SetTitle("cluster size");
  h->GetXaxis()->SetTitle("Layer");
  h->GetXaxis()->SetNdivisions(-10);
  for (int i = 1; i <= nLayers; i++)
    h->GetXaxis()->SetBinLabel(i, name[i - 1]);
  h->SetMaximum(20);
  h->SetMinimum(0);
  h->SetStats(false);
  h->Draw();
  auto grP = new TGraphErrors(nLayers, xP, yP, nullptr, vyP);
  grP->SetMarkerStyle(4);
  grP->SetMarkerSize(2);
  grP->SetTitle("Primary");
  auto grS = new TGraphErrors(nLayers, xS, yS, nullptr, vyS);
  grS->SetMarkerStyle(3);
  grS->SetMarkerSize(2);
  grS->SetTitle("Secondary");
  auto mg = new TMultiGraph("mg", "");
  mg->Add(grP);
  mg->Add(grS);
  mg->Draw("P pmc plc");
  auto leg = new TLegend(0.75, 0.75, 0.9, 0.9);
  leg->AddEntry(grP);
  leg->AddEntry(grS);
  leg->Draw();
  c1->Write();
  c1->SaveAs("its3ClusterSize.pdf");
  for (const auto& hh : {hPrimary, hSecondary, hPionPrimary, hPionSecondary, hProtonPrimary, hProtonSecondary, hKaonPrimary, hKaonSecondary, hOtherPrimary, hOtherSecondary}) {
    for (const auto& h : hh) {
      h.Write();
    }
  }
  for (const auto& hh : {hPrimaryEta, hSecondaryEta, hPionPrimaryEta, hPionSecondaryEta, hProtonPrimaryEta, hProtonSecondaryEta, hKaonPrimaryEta, hKaonSecondaryEta, hOtherPrimaryEta, hOtherSecondaryEta}) {
    for (const auto& h : hh) {
      h.Write();
    }
  }
  for (const auto& hh : {hPrimaryPt, hSecondaryPt, hPionPrimaryPt, hPionSecondaryPt, hProtonPrimaryPt, hProtonSecondaryPt, hKaonPrimaryPt, hKaonSecondaryPt, hOtherPrimaryPt, hOtherSecondaryPt}) {
    for (const auto& h : hh) {
      h.Write();
    }
  }
  for (const auto& hh : {hPrimaryPhi, hSecondaryPhi, hPionPrimaryPhi, hPionSecondaryPhi, hProtonPrimaryPhi, hProtonSecondaryPhi, hKaonPrimaryPhi, hKaonSecondaryPhi, hOtherPrimaryPhi, hOtherSecondaryPhi}) {
    for (const auto& h : hh) {
      h.Write();
    }
  }
  hOuterBarrel.Write();
  sw.Stop();
  sw.Print();
}

void checkFile(const std::unique_ptr<TFile>& file)
{
  if (!file || file->IsZombie()) {
    printf("Could not open %s!\n", file->GetName());
    std::exit(1);
  }
}
