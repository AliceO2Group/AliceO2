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

/// \file CheckTracksCA.C
/// \brief Quality assurance macro for TRK tracking

#if !defined(__CLING__) || defined(__ROOTCLING__)
#include <array>
#include <cmath>
#include <iostream>
#include <unordered_map>
#include <vector>

#include <TFile.h>
#include <TTree.h>
#include <TH1D.h>
#include <TCanvas.h>
#include <THStack.h>
#include <TLegend.h>
#include <TLatex.h>
#include <TStyle.h>

#include "DataFormatsITS/TrackITS.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/MCTrack.h"
#include "Steer/MCKinematicsReader.h"
#include "TRKSimulation/Hit.h"
#include "TRKBase/GeometryTGeo.h"
#include "DetectorsBase/GeometryManager.h"

#endif

using namespace std;
using namespace o2;

/// Structure to track particle hit information
struct ParticleHitInfo {
  std::bitset<11> layerHits; ///< Which layers have hits (11 layers for TRK)
  int nHits = 0;             ///< Total number of hits
  float pt = 0.0f;           ///< Particle pT

  void addHit(int layer)
  {
    if (!layerHits[layer]) {
      layerHits[layer] = true;
      nHits++;
    }
  }

  bool hasConsecutiveLayers(int nConsecutive) const
  {
    for (int startLayer = 0; startLayer <= 11 - nConsecutive; ++startLayer) {
      bool allSet = true;
      for (int i = 0; i < nConsecutive; ++i) {
        if (!layerHits[startLayer + i]) {
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

void CheckTracksCA(std::string tracfile = "o2trac_trk.root",
                   std::string kinefile = "o2sim_Kine.root",
                   std::string hitsfile = "o2sim_HitsTRK.root",
                   std::string outputfile = "trk_qa_output.root")
{
  gStyle->SetOptStat(0);

  std::cout << "=== Starting TRK Track Quality Assurance ===" << std::endl;
  std::cout << "Input files:" << std::endl;
  std::cout << "  Tracks:      " << tracfile << std::endl;
  std::cout << "  Kinematics:  " << kinefile << std::endl;
  std::cout << "  Hits:        " << hitsfile << std::endl;
  std::cout << "  Output:      " << outputfile << std::endl;
  std::cout << std::endl;

  // MC kinematics reader
  o2::steer::MCKinematicsReader kineReader("o2sim", o2::steer::MCKinematicsReader::Mode::kMCKine);
  const int nEvents = kineReader.getNEvents(0);
  std::cout << "Number of MC events: " << nEvents << std::endl;

  // Open hits file to count hits per particle per layer
  TFile* hitsFile = TFile::Open(hitsfile.c_str(), "READ");
  if (!hitsFile || hitsFile->IsZombie()) {
    std::cerr << "ERROR: Cannot open hits file: " << hitsfile << std::endl;
    return;
  }
  TTree* hitsTree = hitsFile->Get<TTree>("o2sim");
  if (!hitsTree) {
    std::cerr << "ERROR: Cannot find o2sim tree in hits file" << std::endl;
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

  // Analyze hits tree to count hits per particle per layer
  std::cout << "Analyzing hits from tree..." << std::endl;
  std::unordered_map<o2::MCCompLabel, ParticleHitInfo> particleHitMap;

  // Load geometry for layer determination
  o2::base::GeometryManager::loadGeometry();
  auto* gman = o2::trk::GeometryTGeo::Instance();

  // Array to map detector to starting layer
  constexpr std::array<int, 2> startLayer{0, 3};

  std::vector<o2::trk::Hit>* trkHit = nullptr;
  hitsTree->SetBranchAddress("TRKHit", &trkHit);

  Long64_t nHitsEntries = hitsTree->GetEntries();
  std::cout << "Processing " << nHitsEntries << " hit entries..." << std::endl;

  for (Long64_t iEntry = 0; iEntry < nHitsEntries; ++iEntry) {
    hitsTree->GetEntry(iEntry);

    for (const auto& hit : *trkHit) {
      // Skip disk hits (only barrel)
      if (gman->getDisk(hit.GetDetectorID()) != -1) {
        continue;
      }

      // Determine layer
      int subDetID = gman->getSubDetID(hit.GetDetectorID());
      const int layer = startLayer[subDetID] + gman->getLayer(hit.GetDetectorID());

      // Create label for this particle
      o2::MCCompLabel label(hit.GetTrackID(), static_cast<int>(iEntry), 0);

      // Add hit to particle's hit map
      particleHitMap[label].addHit(layer);
    }
  }

  std::cout << "Found " << particleHitMap.size() << " unique particles with hits" << std::endl;

  // Store particle info and fill generated histograms
  std::unordered_map<o2::MCCompLabel, float> particlePtMap;

  // Create histograms
  constexpr int nLayers = 11;
  constexpr int nb = 100;
  double xbins[nb + 1], ptcutl = 0.05, ptcuth = 10.;
  double a = std::log(ptcuth / ptcutl) / nb;
  for (int i = 0; i <= nb; i++)
    xbins[i] = ptcutl * std::exp(i * a);

  TH1D genParticlePtHist("genParticlePt", "Generated Particle p_{T} (All Layers); #it{p}_{T} (GeV/#it{c}); Counts", nb, xbins);
  TH1D genParticlePt7LayersHist("genParticlePt7Layers", "Generated Particle p_{T} with hits in at least 7 consecutive layers; #it{p}_{T} (GeV/#it{c}); Counts", nb, xbins);
  TH1D goodTracks("goodTracks", "Good Tracks; p_{T} (GeV/c); Counts", nb, xbins);
  TH1D fakeTracks("fakeTracks", "Fake Tracks; p_{T} (GeV/c); Counts", nb, xbins);

  std::array<TH1D, 5> goodTracksMatching, fakeTracksMatching;
  for (int i = 0; i < 5; ++i) {
    goodTracksMatching[i] = TH1D(Form("goodTracksMatching_%dLayers", i + 7),
                                 Form("Good Tracks with %d layer hits; p_{T} (GeV/c); Counts", i + 7),
                                 nb, xbins);
    fakeTracksMatching[i] = TH1D(Form("fakeTracksMatching_%dLayers", i + 7),
                                 Form("Fake Tracks with %d layer hits; p_{T} (GeV/c); Counts", i + 7),
                                 nb, xbins);
  }

  TH1D numberOfClustersPerTrack("numberOfClustersPerTrack",
                                "Number of clusters per track; N_{clusters}; Counts",
                                12, -0.5, 11.5);

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

      // Store particle info
      particlePtMap[label] = pt;

      // Check if this particle has hits
      auto hitIt = particleHitMap.find(label);
      if (hitIt != particleHitMap.end()) {
        // Store pT in hit info
        hitIt->second.pt = pt;

        // Fill histogram for particles with hits in all 11 layers
        if (hitIt->second.nHits == 11) {
          genParticlePtHist.Fill(pt);
        }

        // Fill histogram for particles with at least 7 consecutive layer hits
        if (hitIt->second.hasConsecutiveLayers(7)) {
          genParticlePt7LayersHist.Fill(pt);
        }
      }
    }
  }

  std::cout << "Generated particles with 11 hits: " << genParticlePtHist.GetEntries() << std::endl;
  std::cout << "Generated particles with 7+ consecutive hits: " << genParticlePt7LayersHist.GetEntries() << std::endl;

  // Second pass: analyze reconstructed tracks
  std::cout << "Analyzing reconstructed tracks..." << std::endl;
  int nROFs = recTree->GetEntries();
  int totalTracks = 0;
  int goodTracksCount = 0;
  int fakeTracksCount = 0;

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

      float pt = mcTracks[trackID].GetPt();

      // Fill histograms
      numberOfClustersPerTrack.Fill(nClusters);

      auto key = o2::MCCompLabel(trackID, eventID, 0);
      if (particleHitMap.find(key) != particleHitMap.end() && particleHitMap[key].hasConsecutiveLayers(11)) {
        if (label.isFake()) {
          fakeTracks.Fill(pt);
          fakeTracksCount++;
          if (nClusters >= 7 && nClusters <= 11) {
            fakeTracksMatching[nClusters - 7].Fill(pt);
          }
        } else {
          goodTracks.Fill(pt);
          goodTracksCount++;
          if (nClusters >= 7 && nClusters <= 11) {
            goodTracksMatching[nClusters - 7].Fill(pt);
          }
        }
      }
    }
  }

  // Create efficiency histograms
  std::cout << "Computing efficiencies..." << std::endl;

  std::array<TH1D, 5> efficiencyHistograms;
  THStack* efficiencyStack = new THStack("efficiencyStack",
                                         "Tracking Efficiency; #it{p}_{T} (GeV/#it{c}); Efficiency");

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
  }

  // Write output
  std::cout << "Writing output to " << outputfile << std::endl;
  TFile outFile(outputfile.c_str(), "RECREATE");
  genParticlePtHist.Write();
  goodTracks.Write();
  fakeTracks.Write();
  for (int i = 0; i < 5; ++i) {
    goodTracksMatching[i].Write();
    fakeTracksMatching[i].Write();
    efficiencyHistograms[i].Write();
  }
  efficiencyStack->Write();
  genParticlePt7LayersHist.Write();
  numberOfClustersPerTrack.Write();
  outFile.Close();

  // Clean up
  hitsFile->Close();
  tracFile->Close();
  delete efficiencyStack;
  delete hitsFile;
  delete tracFile;
}
