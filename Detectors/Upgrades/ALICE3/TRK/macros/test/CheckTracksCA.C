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

/// \author A. Daribayeva
/// Quality assurance test on reconstructed tracks, producing efficiency plots and performance table

#if !defined(__CLING__) || defined(__ROOTCLING__)
#include <array>
#include <cmath>
#include <iostream>
#include <unordered_map>
#include <vector>
#include <fstream>
#include <algorithm>
#include <iomanip>

#include <TFile.h>
#include <TTree.h>
#include <TH1F.h>
#include <TH2D.h>
#include <TCanvas.h>
#include <THStack.h>
#include <TLegend.h>
#include <TLatex.h>
#include <TStyle.h>
#include <TObjArray.h>
#include <TSystem.h>
#include <TROOT.h>
#include <TAxis.h>
#include <TH1.h>

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

enum class RangeMode {
  ContentOnly,
  ContentOrError,
  ReferenceContent
};

void setAutoXRange(TH1* h,
                   RangeMode mode = RangeMode::ContentOnly,
                   const TH1* hRef = nullptr,
                   double threshold = 0.0,
                   int marginBins = 1);

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

bool hasConsecutiveLayers(const o2::its::TrackITS& recoTrack, int nClusters)
{
  std::array<bool, 11> layers{};

  for (int i = 0; i < 11; i++) {
    layers[i] = recoTrack.hasHitOnLayer(i);
  }

  return std::search_n(layers.begin(), layers.end(), nClusters, true) != layers.end();
}

void CheckTracksCA(std::string trackfile = "o2trac_trk.root",
                   std::string kinefile = "o2sim_Kine.root",
                   std::string hitsfile = "o2sim_HitsTRK.root",
                   std::string outFile = "RecoPerformanceTable.dat",
                   std::string outFile1 = "RecoTracksQA.root")
{

  std::cout << "=== Starting TRK Track Reconstruction Quality Assurance ===" << std::endl;
  std::cout << "Input files:" << std::endl;
  std::cout << "  Tracks:      " << trackfile << std::endl;
  std::cout << "  Kinematics:  " << kinefile << std::endl;
  std::cout << "  Hits:        " << hitsfile << std::endl;
  std::cout << "  Output file with performance table: " << outFile << std::endl;
  std::cout << "  Output root file with histograms: " << outFile1 << std::endl;

  gROOT->SetBatch(true);

  // MC kinematics reader
  o2::steer::MCKinematicsReader kineReader("o2sim", o2::steer::MCKinematicsReader::Mode::kMCKine);
  const int nEvents = kineReader.getNEvents(0);

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
  TFile* tracFile = TFile::Open(trackfile.c_str(), "READ");
  if (!tracFile || tracFile->IsZombie()) {
    std::cerr << "ERROR: Cannot open tracks file: " << trackfile << std::endl;
    return;
  }

  TTree* recTree = tracFile->Get<TTree>("o2sim");
  if (!recTree) {
    std::cerr << "ERROR: Cannot find o2sim tree in tracks file" << std::endl;
    return;
  }

  // ============== MC part ===============================
  // Analyze hits tree to count hits per particle per layer
  std::cout << "Analyzing hits from tree..." << std::endl;
  std::unordered_map<o2::MCCompLabel, ParticleHitInfo> particleHitMap;

  // Load geometry for layer determination
  o2::base::GeometryManager::loadGeometry();
  auto* gman = o2::trk::GeometryTGeo::Instance();

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
      const int layer = gman->getBarrelLayer(hit.GetDetectorID());

      // Create label for this particle
      o2::MCCompLabel label(hit.GetTrackID(), static_cast<int>(iEntry), 0);

      // Add hit to particle's hit map
      particleHitMap[label].addHit(layer);
    }
  }

  std::cout << "Found " << particleHitMap.size() << " unique particles with hits" << std::endl;

  //=========== need to set the min and max ranges for hists
  std::vector<float> pTDist;
  std::unordered_map<o2::MCCompLabel, o2::MCTrack> MCTrackMap;

  // counters for general statisics
  int counterPrimaries{0}, counterSecondaries{0};

  for (int iEvent = 0; iEvent < nEvents; ++iEvent) {
    const auto& mcTracks = kineReader.getTracks(iEvent);

    for (size_t iTrack = 0; iTrack < mcTracks.size(); ++iTrack) {
      const auto& mcTrack = mcTracks[iTrack];

      // Create label for this particle
      o2::MCCompLabel label(iTrack, iEvent, 0);

      auto hitIt = particleHitMap.find(label);
      if (hitIt != particleHitMap.end()) {

        if (mcTrack.isPrimary()) {
          counterPrimaries++;
        }
        if (mcTrack.isSecondary()) {
          counterSecondaries++;
        }

        MCTrackMap.emplace(label, mcTrack);
        pTDist.push_back(mcTrack.GetPt());
      }
    }
  }

  int nBins = 100;
  auto [minpT, maxpT] = std::minmax_element(pTDist.begin(), pTDist.end());

  //=========== histograms =============
  // for exclusive studies
  TH1F hPtDenExclusive("", "", nBins, *minpT, *maxpT);

  TH1F hPtNumExclusive("", "", nBins, *minpT, *maxpT);

  TH1F hPtEffExclusive("hPtEffExclusive",
                       "efficiency (exclusive, good, primaries) vs p_{T}; p_{T} [GeV/c]; Efficiency",
                       nBins, *minpT, *maxpT);

  // for inclusive studies
  TH1F hPtDenInclusive("", "", nBins, *minpT, *maxpT);

  TH1F hPtNumInclusive("", "", nBins, *minpT, *maxpT);

  TH1F hPtEffInclusive("hPtEffInclusive", "", nBins, *minpT, *maxpT);

  // for inclusive studies, fake
  TH1F hPtNumInclusiveFake("", "", nBins, *minpT, *maxpT);
  TH1F hPtEffInclusiveFake("", "", nBins, *minpT, *maxpT);

  TH2D hPtResVsPt("", "", nBins, *minpT, *maxpT, 100, -0.5, 0.5);

  // for inclusive efficiencies
  int counterAll{0}, prim_ge7{0}, sec_ge7{0};

  // for exclusive studies when we have 7,8,9,10,11 hits
  std::array<int, 12> mcExact{};

  for (const auto& [label, mcTrack] : MCTrackMap) {

    const auto& hitInfo = particleHitMap.at(label);
    int nHits = hitInfo.nHits;

    if (nHits < 7 || nHits > 11) {
      continue;
    }

    float pT = mcTrack.GetPt();

    bool consecutive7 = hitInfo.hasConsecutiveLayers(7);

    if (mcTrack.isPrimary()) {

      // exclusive - all hits should be on subsequent layers
      if (hitInfo.hasConsecutiveLayers(nHits)) {
        hPtDenExclusive.Fill(pT);
        ++mcExact[nHits];
      }

      // inclusive - it's enough to be on 7 consequtive layers
      if (consecutive7) {
        hPtDenInclusive.Fill(pT);
        ++prim_ge7;
      }

    } else if (mcTrack.isSecondary()) {

      if (consecutive7) {
        ++sec_ge7;
      }
    }
  }

  counterAll = prim_ge7 + sec_ge7;

  //============  reco tracks ===============

  // Reconstructed tracks and labels
  std::vector<o2::its::TrackITS>* recTracks = nullptr;
  std::vector<o2::MCCompLabel>* trkLabels = nullptr;
  std::vector<float> pTResVector; // good, primaries, inclusive

  recTree->SetBranchAddress("TRKTrack", &recTracks);
  recTree->SetBranchAddress("TRKTrackMCTruth", &trkLabels);

  // Second pass: analyze reconstructed tracks
  std::cout << "Analyzing reconstructed tracks..." << std::endl;
  int nROFs = recTree->GetEntries();
  int totalTracks{0};

  // inclusive count
  std::unordered_set<o2::MCCompLabel> foundAllGood, foundAllFake;
  std::unordered_set<o2::MCCompLabel> foundPrimGood, foundPrimFake;
  std::unordered_set<o2::MCCompLabel> foundSecGood, foundSecFake;

  // exclusive count
  std::array<std::unordered_set<o2::MCCompLabel>, 12> foundExclusiveGood, foundExclusiveFake;
  std::array<std::unordered_set<o2::MCCompLabel>, 12> foundWithLessClusters;

  int count7RecoGood{0};

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

      int nClusters = track.getNumberOfClusters();
      if (nClusters < 7 || nClusters > 11) {
        continue;
      }

      auto key = o2::MCCompLabel(label.getTrackID(), label.getEventID(), 0);

      auto hitIt = particleHitMap.find(key);
      auto mcIt = MCTrackMap.find(key);

      if (hitIt == particleHitMap.end() || mcIt == MCTrackMap.end()) {
        continue;
      }

      int nHits = hitIt->second.nHits;
      if (nHits < 7 || nHits > 11) {
        continue;
      }

      bool mcHasN = hitIt->second.hasConsecutiveLayers(nHits);
      bool recoHasN = hasConsecutiveLayers(track, nClusters);

      float mcPt = mcIt->second.GetPt();
      float recoPT = track.getPt();

      // inclusive count
      if (hitIt->second.hasConsecutiveLayers(7) && hasConsecutiveLayers(track, 7)) {

        // for good tracks
        if (label.isCorrect()) {
          foundAllGood.insert(key);

          if (mcIt->second.isPrimary()) {
            foundPrimGood.insert(key);
            hPtNumInclusive.Fill(mcPt);

            float ptRes = (recoPT - mcPt) / mcPt;
            pTResVector.push_back(ptRes);
            hPtResVsPt.Fill(mcPt, ptRes);

          } else if (mcIt->second.isSecondary()) {
            foundSecGood.insert(key);
          }
        }

        // for fake tracks
        if (label.isFake()) {
          foundAllFake.insert(key);

          if (mcIt->second.isPrimary()) {
            foundPrimFake.insert(key);
            hPtNumInclusiveFake.Fill(mcPt);
          } else if (mcIt->second.isSecondary()) {
            foundSecFake.insert(key);
          }
        }
      }

      // exclusive count
      if (nHits == nClusters && mcHasN && recoHasN) {

        if (mcIt->second.isPrimary()) {

          if (label.isCorrect()) {

            hPtNumExclusive.Fill(mcPt);
            foundExclusiveGood[nHits].insert(key);
          }

          if (label.isFake()) {
            foundExclusiveFake[nHits].insert(key);
          }
        }
      }

      // counting cluster loss
      if (mcIt->second.isPrimary() && mcHasN && recoHasN &&
          label.isCorrect() &&
          nClusters < nHits) {

        foundWithLessClusters[nHits].insert(key);
      }

    } // end loop over reco tracks
  } // end loop over RoFs

  // inclusive efficiencies for Good tracks
  float effForAllGood = counterAll > 0 ? 100.f * foundAllGood.size() / counterAll : 0.f;
  float effForPrimGood = prim_ge7 > 0 ? 100.f * foundPrimGood.size() / prim_ge7 : 0.f;
  float effForSecGood = sec_ge7 > 0 ? 100.f * foundSecGood.size() / sec_ge7 : 0.f;

  // inclusive efficiencies for Fake tracks
  float effForAllFake = counterAll > 0 ? 100.f * foundAllFake.size() / counterAll : 0.f;
  float effForPrimFake = prim_ge7 > 0 ? 100.f * foundPrimFake.size() / prim_ge7 : 0.f;
  float effForSecFake = sec_ge7 > 0 ? 100.f * foundSecFake.size() / sec_ge7 : 0.f;

  // exclusive efficiencies for Good and Fake tracks
  std::array<float, 12> effExactAllGood{}, effExactAllFake{};

  for (int n = 7; n <= 11; ++n) {
    effExactAllGood[n] = mcExact[n] > 0 ? 100.f * foundExclusiveGood[n].size() / mcExact[n] : 0.f;
    effExactAllFake[n] = mcExact[n] > 0 ? 100.f * foundExclusiveFake[n].size() / mcExact[n] : 0.f;
  }

  // cluster loss
  std::array<float, 12> fracWithLessClusters{};
  for (int n = 7; n <= 11; ++n) {
    fracWithLessClusters[n] = mcExact[n] > 0 ? 100.f * foundWithLessClusters[n].size() / mcExact[n] : 0.f;
  }

  // pT vs inclusive & exclusive track efficiencies
  hPtEffExclusive.Divide(&hPtNumExclusive, &hPtDenExclusive, 1.0, 1.0, "B");
  hPtEffInclusive.Divide(&hPtNumInclusive, &hPtDenInclusive, 1.0, 1.0, "B");
  hPtEffInclusiveFake.Divide(&hPtNumInclusiveFake, &hPtDenInclusive, 1.0, 1.0, "B");

  // pT resolution for good inclusive tracks, primaries
  auto [minPtRes, maxPtRes] = std::minmax_element(pTResVector.begin(), pTResVector.end());
  TH1F pTResolution("pTResolutionForInclusive", "p_{T} resolution; (p_{T}^{rec}-p_{T}^{MC})/p_{T}^{MC}; Counts", nBins, *minPtRes, *maxPtRes);
  for (const auto& pTVal : pTResVector) {
    pTResolution.Fill(pTVal);
  }
  pTResolution.Fit("gaus");

  TObjArray fitSlices;
  hPtResVsPt.FitSlicesY(nullptr, 0, -1, 0, "QNR", &fitSlices);

  TH1D* hSigmaVsPt = nullptr;

  if (fitSlices.GetEntries() > 2 && fitSlices.At(2)) {
    hSigmaVsPt = dynamic_cast<TH1D*>(fitSlices.At(2)->Clone("hSigmaVsPt"));
    if (hSigmaVsPt) {
      hSigmaVsPt->SetTitle("#sigma(p_{T} resolution) vs p_{T}; p_{T}^{MC} [GeV/c]; #sigma");
      hSigmaVsPt->GetXaxis()->SetRangeUser(0.5, *maxpT);
    }
  }

  // Style
  hPtEffInclusive.SetLineColor(kBlue + 1);
  hPtEffInclusive.SetMarkerColor(kBlue + 1);
  hPtEffInclusive.SetMarkerStyle(20);
  hPtEffInclusive.SetMarkerSize(1.0);
  hPtEffInclusive.SetLineWidth(2);

  hPtEffInclusiveFake.SetLineColor(kRed + 1);
  hPtEffInclusiveFake.SetMarkerColor(kRed + 1);
  hPtEffInclusiveFake.SetMarkerStyle(24);
  hPtEffInclusiveFake.SetMarkerSize(1.0);
  hPtEffInclusiveFake.SetLineWidth(2);

  // Titles and axis labels
  hPtEffInclusive.SetTitle("Inclusive tracking performance vs p_{T}");
  hPtEffInclusive.GetXaxis()->SetTitle("p_{T} [GeV/c]");
  hPtEffInclusive.GetYaxis()->SetTitle("Rate");

  // Canvas
  TCanvas* cPtEff = new TCanvas("", "", 900, 700);

  setAutoXRange(&hPtEffInclusive, RangeMode::ReferenceContent, &hPtDenInclusive);
  setAutoXRange(&hPtEffInclusiveFake, RangeMode::ReferenceContent, &hPtDenInclusive);

  hPtEffInclusive.Draw("E1");
  hPtEffInclusiveFake.Draw("E1 SAME");

  // Legend
  TLegend* leg = new TLegend(0.60, 0.15, 0.88, 0.35);
  leg->SetBorderSize(0);
  leg->SetFillStyle(0);
  leg->AddEntry(&hPtEffInclusive, "Inclusive good efficiency", "lp");
  leg->AddEntry(&hPtEffInclusiveFake, "Inclusive fake rate", "lp");
  leg->Draw("E1 SAME");

  setAutoXRange(&hPtEffExclusive, RangeMode::ContentOnly);

  // Writing to output Root file
  std::cout << "Writing histograms to " << outFile1 << std::endl;
  TFile outFileRoot(outFile1.c_str(), "RECREATE");
  if (hSigmaVsPt) {
    hSigmaVsPt->Write();
  }
  hPtEffExclusive.Write();
  hPtEffInclusive.Write();
  cPtEff->Write();
  pTResolution.Write();
  outFileRoot.Close();

  // Building performance table
  std::cout << "Building performance table ... " << std::endl;
  std::ofstream outFileTxt(outFile.c_str());
  outFileTxt << std::fixed << std::setprecision(2);

  outFileTxt << "This is preliminary reconstruction performance table !!" << std::endl;
  outFileTxt << "\nGenerated " << particleHitMap.size() << " unique particles with hits" << std::endl;
  outFileTxt << "Among them, N primaries: " << counterPrimaries << " and secondaries: " << counterSecondaries << std::endl;
  outFileTxt << "Number of total reconstructed tracks: " << totalTracks << std::endl;

  outFileTxt << "\nReconstruction performance table\n\n";

  outFileTxt << "| "
             << std::left << std::setw(20) << "Track category"
             << "| " << std::setw(14) << "Efficiency (%)"
             << "| " << std::setw(14) << "Fake rate (%)"
             << "| " << std::setw(12) << "MC counts"
             << " |\n";

  outFileTxt << std::string(70, '-') << "\n";

  outFileTxt << "| "
             << std::left << std::setw(20) << "All (prim+sec)"
             << "| " << std::setw(14) << effForAllGood
             << "| " << std::setw(14) << effForAllFake
             << "| " << std::setw(12) << counterAll
             << " |\n";

  outFileTxt << "| "
             << std::left << std::setw(20) << "Primaries"
             << "| " << std::setw(14) << effForPrimGood
             << "| " << std::setw(14) << effForPrimFake
             << "| " << std::setw(12) << prim_ge7
             << " |\n";

  outFileTxt << "| "
             << std::left << std::setw(20) << "Secondaries"
             << "| " << std::setw(14) << effForSecGood
             << "| " << std::setw(14) << effForSecFake
             << "| " << std::setw(12) << sec_ge7
             << " |\n";

  outFileTxt << "\n\nExclusive efficiencies for primaries:\n\n";

  outFileTxt << "| "
             << std::left << std::setw(15) << "Track length"
             << "| " << std::setw(14) << "Efficiency (%)"
             << "| " << std::setw(14) << "Fake rate (%)"
             << "| " << std::setw(14) << "Cluster loss (%)"
             << "| " << std::setw(14) << "MC counts"
             << " |\n";

  outFileTxt << std::string(85, '-') << "\n";

  for (int n = 11; n >= 7; --n) {
    outFileTxt << "| "
               << std::left << std::setw(15) << (std::to_string(n) + "-hit")
               << "| " << std::setw(14) << effExactAllGood[n]
               << "| " << std::setw(14) << effExactAllFake[n]
               << "| " << std::setw(16) << fracWithLessClusters[n]
               << "| " << std::setw(14) << mcExact[n]
               << " |\n";
  }

  std::cout << "Analysis complete!" << std::endl;

} // end of macro

void setAutoXRange(TH1* h, RangeMode mode,
                   const TH1* hRef,
                   double threshold,
                   int marginBins)
{
  if (!h)
    return;

  const TH1* hScan = h;

  if (mode == RangeMode::ReferenceContent) {
    if (!hRef)
      return;
    hScan = hRef;
  }

  const int nBins = hScan->GetNbinsX();
  int first = -1;
  int last = -1;

  auto isUsefulBin = [&](int i) -> bool {
    const double content = hScan->GetBinContent(i);
    const double error = hScan->GetBinError(i);

    switch (mode) {
      case RangeMode::ContentOnly:
        return content > threshold;

      case RangeMode::ContentOrError:
        return (content > threshold) || (error > 0.0);

      case RangeMode::ReferenceContent:
        return content > threshold;
    }
    return false;
  };

  for (int i = 1; i <= nBins; ++i) {
    if (isUsefulBin(i)) {
      first = i;
      break;
    }
  }

  for (int i = nBins; i >= 1; --i) {
    if (isUsefulBin(i)) {
      last = i;
      break;
    }
  }

  if (first == -1 || last == -1 || first > last) {
    return;
  }

  first = std::max(1, first - marginBins);
  last = std::min(nBins, last + marginBins);

  h->GetXaxis()->SetRange(first, last);
}
