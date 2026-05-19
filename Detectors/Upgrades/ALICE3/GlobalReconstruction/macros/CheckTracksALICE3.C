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

/// \file CheckTracksALICE3.C
/// \brief Quality assurance macro for TRK tracking

#if !defined(__CLING__) || defined(__ROOTCLING__)
#include <array>
#include <cmath>
#include <iostream>
#include <unordered_map>
#include <vector>
#include <unordered_set>

#include <TFile.h>
#include <TTree.h>
#include <TH1D.h>
#include <TCanvas.h>
#include <THStack.h>
#include <TLegend.h>
#include <TLatex.h>
#include <TStyle.h>

#include "DataFormatsITS/TrackITS.h"
#include "DataFormatsTRK/Cluster.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/MCTrack.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "SimulationDataFormat/O2DatabasePDG.h"
#include "Steer/MCKinematicsReader.h"

#endif

using namespace std;
using namespace o2;

struct ParticleClusterInfo {
  std::bitset<11> layerClusters;
  int nClusters = 0;
  float pt = 0.0f;

  void addCluster(int layer)
  {
    if (!layerClusters[layer]) {
      layerClusters[layer] = true;
      nClusters++;
    }
  }

  bool hasConsecutiveLayers(int nConsecutive) const
  {
    for (int startLayer = 0; startLayer <= 11 - nConsecutive; ++startLayer) {
      bool allSet = true;
      for (int i = 0; i < nConsecutive; ++i) {
        if (!layerClusters[startLayer + i]) {
          allSet = false;
          break;
        }
      }
      if (allSet) {
        return true;
      }
    }
    return false;
  }
};

void CheckTracksALICE3(std::string tracfile = "o2trac_trk.root",
                       std::string simprefix = "o2sim",
                       std::string clusfile = "o2clus_trk.root",
                       std::string outputfile = "trk_qa_output.root")
{
  gStyle->SetOptStat(0);

  std::cout << "=== Starting TRK Track Quality Assurance ===" << std::endl;
  std::cout << "Input files:" << std::endl;
  std::cout << "  Tracks:      " << tracfile << std::endl;
  std::cout << "  Sim prefix:  " << simprefix << std::endl;
  std::cout << "  Clusters:    " << clusfile << std::endl;
  std::cout << "  Output:      " << outputfile << std::endl;
  std::cout << std::endl;

  // MC kinematics reader
  o2::steer::MCKinematicsReader kineReader(simprefix, o2::steer::MCKinematicsReader::Mode::kMCKine);
  const int nEvents = kineReader.getNEvents(0);
  std::cout << "Number of MC events: " << nEvents << std::endl;

  // Open clusters file to count cluster-associated layers per particle
  TFile* clustersFile = TFile::Open(clusfile.c_str(), "READ");
  if (!clustersFile || clustersFile->IsZombie()) {
    std::cerr << "ERROR: Cannot open clusters file: " << clusfile << std::endl;
    return;
  }
  TTree* clusTree = clustersFile->Get<TTree>("o2sim");
  if (!clusTree) {
    std::cerr << "ERROR: Cannot find o2sim tree in clusters file" << std::endl;
    return;
  }

  // Open reconstructed tracks file
  TFile* tracFile = TFile::Open(tracfile.c_str(), "READ");
  if (!tracFile || tracFile->IsZombie()) {
    std::cerr << "ERROR: Cannot open tracks file: " << tracfile << std::endl;
    return;
  }
  TTree* recTree = tracFile->Get<TTree>("o2sim");
  if (!recTree) {
    std::cerr << "ERROR: Cannot find o2sim tree in tracks file" << std::endl;
    return;
  }

  // Reconstructed tracks and labels
  std::vector<o2::its::TrackITS>* recTracks = nullptr;
  std::vector<o2::MCCompLabel>* trkLabels = nullptr;
  recTree->SetBranchAddress("TRKTrack", &recTracks);
  recTree->SetBranchAddress("TRKTrackMCTruth", &trkLabels);

  std::cout << "Reading tracks from tree..." << std::endl;

  // Analyze cluster tree to count cluster-associated layers per particle
  std::cout << "Analyzing clusters from tree..." << std::endl;
  std::unordered_map<o2::MCCompLabel, ParticleClusterInfo> particleClusterMap;

  static constexpr int nTRKLayers = 11;
  std::array<std::vector<o2::trk::Cluster>*, nTRKLayers> clustersPerLayer{};
  std::array<o2::dataformats::MCTruthContainer<o2::MCCompLabel>*, nTRKLayers> clusterLabelsPerLayer{};

  for (int iLayer = 0; iLayer < nTRKLayers; ++iLayer) {
    const std::string clusBranch = std::string("TRKClusterComp_") + std::to_string(iLayer);
    const std::string truthBranch = std::string("TRKClusterMCTruth_") + std::to_string(iLayer);
    if (!clusTree->GetBranch(clusBranch.c_str())) {
      std::cerr << "WARNING: Missing cluster branch for layer " << iLayer << " (expected " << clusBranch << ")" << std::endl;
      continue;
    }
    if (!clusTree->GetBranch(truthBranch.c_str())) {
      std::cerr << "WARNING: Missing cluster MC-truth branch for layer " << iLayer << " (expected " << truthBranch << ")" << std::endl;
      continue;
    }
    clusTree->SetBranchAddress(clusBranch.c_str(), &clustersPerLayer[iLayer]);
    clusTree->SetBranchAddress(truthBranch.c_str(), &clusterLabelsPerLayer[iLayer]);
  }

  Long64_t nClusEntries = clusTree->GetEntries();
  std::cout << "Processing " << nClusEntries << " cluster entries..." << std::endl;

  for (Long64_t iEntry = 0; iEntry < nClusEntries; ++iEntry) {
    clusTree->GetEntry(iEntry);
    for (int iLayer = 0; iLayer < nTRKLayers; ++iLayer) {
      const auto* clusArr = clustersPerLayer[iLayer];
      const auto* clusLabArr = clusterLabelsPerLayer[iLayer];
      if (!clusArr || !clusLabArr) {
        continue;
      }
      for (size_t iClus = 0; iClus < clusArr->size(); ++iClus) {
        const auto labels = clusLabArr->getLabels(iClus);
        if (labels.empty()) {
          continue;
        }
        const auto& lab = labels[0];
        if (!lab.isValid() || lab.getSourceID() != 0 || !lab.isCorrect()) {
          continue;
        }
        int trackID = -1, evID = -1, srcID = -1;
        bool fake = false;
        lab.get(trackID, evID, srcID, fake);
        if (trackID < 0 || evID < 0) {
          continue;
        }
        particleClusterMap[o2::MCCompLabel(trackID, evID, 0)].addCluster(iLayer);
      }
    }
  }

  std::cout << "Found " << particleClusterMap.size() << " unique particles with clusters" << std::endl;

  // Store particle info and fill generated histograms
  std::unordered_map<o2::MCCompLabel, float> particlePtMap;

  // Create histograms
  constexpr int nb = 100;
  double xbins[nb + 1], ptcutl = 0.05, ptcuth = 10.;
  double a = std::log(ptcuth / ptcutl) / nb;
  for (int i = 0; i <= nb; i++)
    xbins[i] = ptcutl * std::exp(i * a);

  TH1D genParticlePtHist("genParticlePt", "Generated Particle p_{T} (All Layers); #it{p}_{T} (GeV/#it{c}); Counts", nb, xbins);
  TH1D genParticlePt7LayersHist("genParticlePt7Layers", "Generated Particle p_{T} with clusters in at least 7 consecutive layers; #it{p}_{T} (GeV/#it{c}); Counts", nb, xbins);
  TH1D chargedPrimaryPtHist("chargedPrimaryPt",
                            "Charged primary particles |#eta| < 2; #it{p}_{T} (GeV/#it{c}); Counts",
                            nb, xbins);
  TH1D goodTracks("goodTracks", "Good Tracks; p_{T} (GeV/c); Counts", nb, xbins);
  TH1D fakeTracks("fakeTracks", "Fake Tracks; p_{T} (GeV/c); Counts", nb, xbins);

  std::array<TH1D, 5> goodTracksMatching, fakeTracksMatching;
  for (int i = 0; i < 5; ++i) {
    goodTracksMatching[i] = TH1D(Form("goodTracksMatching_%dLayers", i + 7),
                                 Form("Good Tracks with %d cluster layers; p_{T} (GeV/c); Counts", i + 7),
                                 nb, xbins);
    fakeTracksMatching[i] = TH1D(Form("fakeTracksMatching_%dLayers", i + 7),
                                 Form("Fake Tracks with %d cluster layers; p_{T} (GeV/c); Counts", i + 7),
                                 nb, xbins);
  }

  TH1D numberOfClustersPerTrack("numberOfClustersPerTrack",
                                "Number of clusters per track; N_{clusters}; Counts",
                                12, -0.5, 11.5);
  TH1D cloneTracks("cloneTracks", "Clone Tracks; #it{p}_{T} (GeV/#it{c}); Counts", nb, xbins);

  std::array<TH1D, 5> duplicateTracksMatching;
  for (int i = 0; i < 5; ++i) {
    duplicateTracksMatching[i] = TH1D(Form("duplicateTracksMatching_%dLayers", i + 7),
                                      Form("Duplicate Tracks with %d cluster layers; p_{T} (GeV/c); Counts", i + 7),
                                      nb, xbins);
  }

  TH1D genParticleEtaHist("genParticleEta",
                          "Generated Particle #eta (11 consec. layers, p_{T} > 1 GeV/c); #eta; Counts",
                          100, -2.5, 2.5);
  std::array<TH1D, 5> goodTracksMatchingEta;
  for (int i = 0; i < 5; ++i) {
    goodTracksMatchingEta[i] = TH1D(Form("goodTracksMatchingEta_%dLayers", i + 7),
                                    Form("Good Tracks #eta with %d cluster layers (p_{T} > 1 GeV/c); #eta; Counts", i + 7),
                                    100, -2.5, 2.5);
  }

  // Numerators for summary efficiency/fake/duplicate vs 7-layer reference
  TH1D goodTracks7("goodTracks7Layers", "Good Tracks (7 consec. layers ref.); #it{p}_{T} (GeV/#it{c}); Counts", nb, xbins);
  TH1D fakeTracks7("fakeTracks7Layers", "Fake Tracks (7 consec. layers ref.); #it{p}_{T} (GeV/#it{c}); Counts", nb, xbins);
  TH1D cloneTracks7("cloneTracks7Layers", "Clone Tracks (7 consec. layers ref.); #it{p}_{T} (GeV/#it{c}); Counts", nb, xbins);

  // Deduplicated fake/clone numerators for 11-layer reference summary
  TH1D fakeTracks11("fakeTracks11Layers", "Fake Tracks (11 consec. layers ref.); #it{p}_{T} (GeV/#it{c}); Counts", nb, xbins);
  TH1D cloneTracks11("cloneTracks11Layers", "Clone Tracks (11 consec. layers ref.); #it{p}_{T} (GeV/#it{c}); Counts", nb, xbins);

  // First pass: identify particles with full hit coverage from kinematics
  std::cout << "Analyzing MC particles..." << std::endl;
  for (int iEvent = 0; iEvent < nEvents; ++iEvent) {
    const auto& mcTracks = kineReader.getTracks(iEvent);
    for (size_t iTrack = 0; iTrack < mcTracks.size(); ++iTrack) {
      const auto& mcTrack = mcTracks[iTrack];
      if (!mcTrack.isPrimary()) {
        continue;
      }

      // Create label for this particle
      o2::MCCompLabel label(iTrack, iEvent, 0);
      float pt = mcTrack.GetPt();

      // Charged primary in |eta| < 2
      if (std::abs(mcTrack.GetEta()) < 2.f) {
        auto* pdgPart = o2::O2DatabasePDG::Instance()->GetParticle(mcTrack.GetPdgCode());
        if (pdgPart != nullptr && pdgPart->Charge() != 0.) {
          chargedPrimaryPtHist.Fill(pt);
        }
      }

      // Store particle info
      particlePtMap[label] = pt;

      auto clusIt = particleClusterMap.find(label);
      if (clusIt != particleClusterMap.end()) {
        clusIt->second.pt = pt;

        if (clusIt->second.hasConsecutiveLayers(11)) {
          genParticlePtHist.Fill(pt);
          if (pt > 1.f) {
            genParticleEtaHist.Fill(mcTrack.GetEta());
          }
        }

        if (clusIt->second.hasConsecutiveLayers(7)) {
          genParticlePt7LayersHist.Fill(pt);
        }
      }
    }
  }

  std::cout << "Generated particles with 11 cluster layers: " << genParticlePtHist.GetEntries() << std::endl;
  std::cout << "Generated particles with 7+ consecutive cluster layers: " << genParticlePt7LayersHist.GetEntries() << std::endl;

  // Count how many reconstructed tracks point to each MC label (clone detection)
  std::unordered_map<o2::MCCompLabel, int> labelRecoCount;
  {
    int nROFsTmp = recTree->GetEntries();
    for (int iROF = 0; iROF < nROFsTmp; ++iROF) {
      recTree->GetEntry(iROF);
      if (!trkLabels) {
        continue;
      }
      for (const auto& lab : *trkLabels) {
        if (!lab.isSet() || !lab.isValid() || lab.isFake()) {
          continue;
        }
        int eventID = lab.getEventID();
        int trackID = lab.getTrackID();
        if (eventID < 0 || eventID >= nEvents) {
          continue;
        }
        const auto& mcTracks = kineReader.getTracks(eventID);
        if (trackID < 0 || trackID >= (int)mcTracks.size()) {
          continue;
        }
        if (!mcTracks[trackID].isPrimary()) {
          continue;
        }
        labelRecoCount[o2::MCCompLabel(lab.getTrackID(), lab.getEventID(), 0)]++;
      }
    }
  }

  // Second pass: analyze reconstructed tracks
  std::cout << "Analyzing reconstructed tracks..." << std::endl;
  int nROFs = recTree->GetEntries();
  int totalTracks = 0;
  int goodTracksCount = 0;
  int fakeTracksCount = 0;
  int cloneTracksCount = 0;
  // Track which MC labels have already been filled per matching bin to avoid double-counting clones
  std::array<std::unordered_set<o2::MCCompLabel>, 5> filledGoodLabels;
  std::unordered_set<o2::MCCompLabel> filledGoodLabelsAny;
  std::unordered_set<o2::MCCompLabel> filledGoodLabelsAny7;
  std::unordered_set<o2::MCCompLabel> filledFakeLabelsAny11;
  std::unordered_set<o2::MCCompLabel> filledCloneLabelsAny11;

  for (int iROF = 0; iROF < nROFs; ++iROF) {
    recTree->GetEntry(iROF);

    if (!recTracks || !trkLabels) {
      continue;
    }

    totalTracks += recTracks->size();

    for (size_t iTrack = 0; iTrack < recTracks->size(); ++iTrack) {
      const auto& track = recTracks->at(iTrack);
      const auto& label = trkLabels->at(iTrack);

      if (!label.isSet() || !label.isValid()) {
        continue;
      }

      int eventID = label.getEventID();
      int trackID = label.getTrackID();
      int nClusters = track.getNumberOfClusters();

      // Get MC track info
      if (eventID < 0 || eventID >= nEvents) {
        continue;
      }

      const auto& mcTracks = kineReader.getTracks(eventID);
      if (trackID < 0 || trackID >= (int)mcTracks.size()) {
        continue;
      }
      if (!mcTracks[trackID].isPrimary()) {
        continue;
      }

      float pt = mcTracks[trackID].GetPt();
      float eta = mcTracks[trackID].GetEta();

      // Fill histograms
      numberOfClustersPerTrack.Fill(nClusters);

      auto key = o2::MCCompLabel(trackID, eventID, 0);
      if (particleClusterMap.find(key) != particleClusterMap.end() && particleClusterMap[key].hasConsecutiveLayers(11)) {
        if (label.isFake()) {
          fakeTracks.Fill(pt);
          fakeTracksCount++;
          if (nClusters >= 7 && nClusters <= 11) {
            fakeTracksMatching[nClusters - 7].Fill(pt);
          }
          filledFakeLabelsAny11.insert(key);
        } else {
          if (filledGoodLabelsAny.insert(key).second) {
            goodTracks.Fill(pt);
            goodTracksCount++;
          }
          if (nClusters >= 7 && nClusters <= 11) {
            int bin = nClusters - 7;
            if (filledGoodLabels[bin].insert(key).second) {
              goodTracksMatching[bin].Fill(pt);
              if (pt > 1.f) {
                goodTracksMatchingEta[bin].Fill(eta);
              }
            } else {
              duplicateTracksMatching[bin].Fill(pt);
            }
          }
          if (labelRecoCount[key] > 1) {
            cloneTracks.Fill(pt);
            cloneTracksCount++;
            filledCloneLabelsAny11.insert(key);
          }
        }
      }

      // Fill summary histograms vs 7-layer reference
      auto clusIt7 = particleClusterMap.find(key);
      if (clusIt7 != particleClusterMap.end() && clusIt7->second.hasConsecutiveLayers(7)) {
        if (label.isFake()) {
          fakeTracks7.Fill(pt);
        } else {
          if (filledGoodLabelsAny7.insert(key).second) {
            goodTracks7.Fill(pt);
          }
          if (labelRecoCount[key] > 1) {
            cloneTracks7.Fill(pt);
          }
        }
      }
    }
  }

  // Create efficiency histograms
  std::cout << "Total tracks: " << totalTracks << ". Out of those matching particles with 11 clusters, good: " << goodTracksCount
            << ", fake: " << fakeTracksCount << ", clones: " << cloneTracksCount << std::endl;

  std::cout << "Computing efficiencies..." << std::endl;

  std::array<TH1D, 5> efficiencyHistograms;
  THStack* efficiencyStack = new THStack("efficiencyStack",
                                         "Tracking Efficiency; #it{p}_{T} (GeV/#it{c}); Efficiency");

  std::array<TH1D, 5> efficiencyEtaHistograms;
  THStack* efficiencyEtaStack = new THStack("efficiencyEtaStack",
                                            "Tracking Efficiency vs #eta (p_{T} > 1 GeV/c); #eta; Efficiency");

  int colors[5] = {kRed, kBlue, kGreen + 2, kMagenta, kOrange};
  for (int i = 0; i < 5; ++i) {
    int nClusters = i + 7;
    efficiencyHistograms[i] = TH1D(Form("efficiency_%dClusters", nClusters),
                                   Form("Efficiency for %d cluster tracks; #it{p}_{T} (GeV/#it{c}); Efficiency", nClusters),
                                   nb, xbins);

    efficiencyHistograms[i].Divide(&goodTracksMatching[i], &genParticlePtHist, 1, 1, "B");

    efficiencyHistograms[i].SetLineColor(colors[i]);
    efficiencyHistograms[i].SetFillColor(colors[i]);
    efficiencyHistograms[i].SetLineWidth(2);
    efficiencyHistograms[i].SetMarkerColor(colors[i]);
    efficiencyHistograms[i].SetMarkerStyle(20 + i);
    efficiencyStack->Add(&efficiencyHistograms[i]);

    efficiencyEtaHistograms[i] = TH1D(Form("efficiencyEta_%dClusters", nClusters),
                                      Form("Efficiency vs #eta for %d cluster tracks (p_{T} > 1 GeV/c); #eta; Efficiency", nClusters),
                                      100, -2.5, 2.5);
    efficiencyEtaHistograms[i].Divide(&goodTracksMatchingEta[i], &genParticleEtaHist, 1, 1, "B");
    efficiencyEtaHistograms[i].SetLineColor(colors[i]);
    efficiencyEtaHistograms[i].SetFillColor(colors[i]);
    efficiencyEtaHistograms[i].SetLineWidth(2);
    efficiencyEtaHistograms[i].SetMarkerColor(colors[i]);
    efficiencyEtaHistograms[i].SetMarkerStyle(20 + i);
    efficiencyEtaStack->Add(&efficiencyEtaHistograms[i]);
  }

  // Build summary efficiency/fake/duplicate vs 7-layer reference
  TH1D effVs7("efficiencyVs7Layers",
              "Tracking Efficiency (7 consec. layers ref.); #it{p}_{T} (GeV/#it{c}); Rate",
              nb, xbins);
  effVs7.Divide(&goodTracks7, &genParticlePt7LayersHist, 1, 1, "B");
  effVs7.SetLineColor(kBlue);
  effVs7.SetLineWidth(2);
  effVs7.SetMarkerColor(kBlue);
  effVs7.SetMarkerStyle(20);

  TH1D fakeVs7("fakeRateVs7Layers",
               "Fake Rate (7 consec. layers ref.); #it{p}_{T} (GeV/#it{c}); Rate",
               nb, xbins);
  fakeVs7.Divide(&fakeTracks7, &genParticlePt7LayersHist, 1, 1, "B");
  fakeVs7.SetLineColor(kRed);
  fakeVs7.SetLineWidth(2);
  fakeVs7.SetMarkerColor(kRed);
  fakeVs7.SetMarkerStyle(21);

  TH1D dupVs7("duplicateRateVs7Layers",
              "Duplicate Rate (7 consec. layers ref.); #it{p}_{T} (GeV/#it{c}); Rate",
              nb, xbins);
  dupVs7.Divide(&cloneTracks7, &genParticlePt7LayersHist, 1, 1, "B");
  dupVs7.SetLineColor(kGreen + 2);
  dupVs7.SetLineWidth(2);
  dupVs7.SetMarkerColor(kGreen + 2);
  dupVs7.SetMarkerStyle(22);

  // Build summary efficiency/fake/duplicate vs 11-layer reference
  // Fill deduplicated fake/clone histograms from the sets collected during the reco loop
  for (const auto& [lbl, info] : particleClusterMap) {
    if (!info.hasConsecutiveLayers(11)) {
      continue;
    }
    auto ptIt = particlePtMap.find(lbl);
    if (ptIt == particlePtMap.end()) {
      continue;
    }
    float ptLbl = ptIt->second;
    if (filledFakeLabelsAny11.count(lbl)) {
      fakeTracks11.Fill(ptLbl);
    }
    if (filledCloneLabelsAny11.count(lbl)) {
      cloneTracks11.Fill(ptLbl);
    }
  }

  TH1D effVs11("efficiencyVs11Layers",
               "Tracking Efficiency (11 consec. layers ref.); #it{p}_{T} (GeV/#it{c}); Rate",
               nb, xbins);
  effVs11.Divide(&goodTracks, &genParticlePtHist, 1, 1, "B");
  effVs11.SetLineColor(kBlue);
  effVs11.SetLineWidth(2);
  effVs11.SetMarkerColor(kBlue);
  effVs11.SetMarkerStyle(20);

  TH1D fakeVs11("fakeRateVs11Layers",
                "Fake Rate (11 consec. layers ref.); #it{p}_{T} (GeV/#it{c}); Rate",
                nb, xbins);
  fakeVs11.Divide(&fakeTracks11, &genParticlePtHist, 1, 1, "B");
  fakeVs11.SetLineColor(kRed);
  fakeVs11.SetLineWidth(2);
  fakeVs11.SetMarkerColor(kRed);
  fakeVs11.SetMarkerStyle(21);

  TH1D dupVs11("duplicateRateVs11Layers",
               "Duplicate Rate (11 consec. layers ref.); #it{p}_{T} (GeV/#it{c}); Rate",
               nb, xbins);
  dupVs11.Divide(&cloneTracks11, &genParticlePtHist, 1, 1, "B");
  dupVs11.SetLineColor(kGreen + 2);
  dupVs11.SetLineWidth(2);
  dupVs11.SetMarkerColor(kGreen + 2);
  dupVs11.SetMarkerStyle(22);

  // Summary canvas — 7-layer reference
  TCanvas summaryCanvas("summaryCanvas7Layers", "TRK Tracking QA Summary (7 layers ref.)", 800, 600);
  summaryCanvas.SetLogx();
  double ymax = std::max({effVs7.GetMaximum(), fakeVs7.GetMaximum(), dupVs7.GetMaximum()});
  effVs7.GetYaxis()->SetRangeUser(0., 1.1 * ymax + 0.05);
  effVs7.Draw("E");
  fakeVs7.Draw("E SAME");
  dupVs7.Draw("E SAME");
  TLegend leg(0.65, 0.70, 0.88, 0.88);
  leg.SetBorderSize(0);
  leg.AddEntry(&effVs7, "Efficiency", "lp");
  leg.AddEntry(&fakeVs7, "Fake rate", "lp");
  leg.AddEntry(&dupVs7, "Duplicate rate", "lp");
  leg.Draw();

  // Summary canvas — 11-layer reference
  TCanvas summaryCanvas11("summaryCanvas11Layers", "TRK Tracking QA Summary (11 layers ref.)", 800, 600);
  summaryCanvas11.SetLogx();
  double ymax11 = std::max({effVs11.GetMaximum(), fakeVs11.GetMaximum(), dupVs11.GetMaximum()});
  effVs11.GetYaxis()->SetRangeUser(0., 1.1 * ymax11 + 0.05);
  effVs11.Draw("E");
  fakeVs11.Draw("E SAME");
  dupVs11.Draw("E SAME");
  TLegend leg11(0.65, 0.70, 0.88, 0.88);
  leg11.SetBorderSize(0);
  leg11.AddEntry(&effVs11, "Efficiency", "lp");
  leg11.AddEntry(&fakeVs11, "Fake rate", "lp");
  leg11.AddEntry(&dupVs11, "Duplicate rate", "lp");
  leg11.Draw();

  // Write output
  std::cout << "Writing output to " << outputfile << std::endl;
  TFile outFile(outputfile.c_str(), "RECREATE");

  // Top-level: summary plots
  summaryCanvas.Write();
  effVs7.Write();
  fakeVs7.Write();
  dupVs7.Write();
  summaryCanvas11.Write();
  effVs11.Write();
  fakeVs11.Write();
  dupVs11.Write();

  // Details directory: per-cluster-count breakdowns and raw counts
  TDirectory* detDir = outFile.mkdir("details");
  detDir->cd();
  genParticlePtHist.Write();
  genParticlePt7LayersHist.Write();
  genParticleEtaHist.Write();
  chargedPrimaryPtHist.Write();
  goodTracks.Write();
  fakeTracks.Write();
  cloneTracks.Write();
  goodTracks7.Write();
  fakeTracks7.Write();
  cloneTracks7.Write();
  fakeTracks11.Write();
  cloneTracks11.Write();
  numberOfClustersPerTrack.Write();
  for (int i = 0; i < 5; ++i) {
    goodTracksMatching[i].Write();
    fakeTracksMatching[i].Write();
    duplicateTracksMatching[i].Write();
    efficiencyHistograms[i].Write();
    goodTracksMatchingEta[i].Write();
    efficiencyEtaHistograms[i].Write();
  }
  efficiencyStack->Write();
  efficiencyEtaStack->Write();

  outFile.Close();

  // Clean up
  clustersFile->Close();
  tracFile->Close();
  delete efficiencyStack;
  delete efficiencyEtaStack;
  delete clustersFile;
  delete tracFile;
}
