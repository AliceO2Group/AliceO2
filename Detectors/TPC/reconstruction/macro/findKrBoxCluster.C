// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file findKrBoxCluster.C
/// \brief This macro retrieves clusters from Krypton and X-Ray runs, input tpcdigits.root
/// \author Philip Hauer <philip.hauer@cern.ch>

#if !defined(__CLING__) || defined(__ROOTCLING__)
#include "TCanvas.h"
#include "TFile.h"
#include "TTree.h"

#include "TPCReconstruction/KrCluster.h"
#include "TPCReconstruction/KrBoxClusterFinder.h"
#include "DataFormatsTPC/Digit.h"

#include <array>
#include <iostream>
#include <tuple>
#include <vector>
#endif

void findKrBoxCluster(int lastTimeBin = 20000, int run = -1, int time = -1, std::string_view gainMapFile = "")
{
  // Read the digits:
  TFile* file = new TFile("data_20210303_0911.digits.root");
  TTree* tree = (TTree*)file->Get("o2sim");
  Long64_t nEntries = tree->GetEntries();
  std::cout << "The Tree has " << nEntries << " Entries." << std::endl;

  // Initialize File for later writing
  TFile* fOut = new TFile("SpeedTest/BoxClusters_setDigitZero.root", "RECREATE");
  TTree* tClusters = new TTree("Clusters", "Clusters");

  // Create a Branch for each sector:
  std::vector<o2::tpc::KrCluster> clusters;
  tClusters->Branch("cls", &clusters);
  tClusters->Branch("run", &run);
  tClusters->Branch("time", &time);

  std::array<std::vector<o2::tpc::Digit>*, 36> digitizedSignal;
  for (size_t iSec = 0; iSec < digitizedSignal.size(); ++iSec) {
    digitizedSignal[iSec] = nullptr;
    tree->SetBranchAddress(Form("TPCDigit_%zu", iSec), &digitizedSignal[iSec]);
  }

  // Create KrBoxClusterFinder object, memory is only allocated once
  auto clFinder = std::make_unique<o2::tpc::KrBoxClusterFinder>();
  if (gainMapFile.size()) {
    clFinder->loadGainMapFromFile(gainMapFile);
  }

  ofstream outFile;
  outFile.open("SpeedTest/2Neighbours_setDigitZero.dat");

  // Now everything can get processed
  // Loop over all events
  for (int iEvent = 0; iEvent < nEntries; ++iEvent) {
    std::cout << iEvent + 1 << "/" << nEntries << std::endl;
    tree->GetEntry(iEvent);
    // Each event consists of sectors (atm only two)

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 36; i++) {
      auto sector = digitizedSignal[i];
      if (sector->size() == 0) {
        continue;
      }

      // Fill map and (if specified) correct with existing gain map
      auto startFillMap = std::chrono::high_resolution_clock::now();
      clFinder->fillAndCorrectMap(*sector, i);
      auto endFillMap = std::chrono::high_resolution_clock::now();
      auto durationFillMap = std::chrono::duration_cast<std::chrono::microseconds>(endFillMap - startFillMap);
      std::cout << "Time to fill map: " << durationFillMap.count() << std::endl;

      // Find all local maxima in sector
      auto startFindMaxima = std::chrono::high_resolution_clock::now();
      std::vector<std::tuple<int, int, int>> localMaxima = clFinder->findLocalMaxima();
      auto endFindMaxima = std::chrono::high_resolution_clock::now();
      auto durationMaxima = std::chrono::duration_cast<std::chrono::microseconds>(endFindMaxima - startFindMaxima);
      std::cout << "Time to find Maxs: " << durationMaxima.count() << std::endl;
      
      // Loop over cluster centers = local maxima
      auto startBuildCluster = std::chrono::high_resolution_clock::now();
      for (const std::tuple<int, int, int>& coords : localMaxima) {
        int padMax = std::get<0>(coords);
        int rowMax = std::get<1>(coords);
        int timeMax = std::get<2>(coords);

        if (timeMax >= lastTimeBin) {
          continue;
        }
        // Build total cluster
        o2::tpc::KrCluster tempCluster = clFinder->buildCluster(padMax, rowMax, timeMax);
        tempCluster.sector = i;

        clusters.emplace_back(tempCluster);
      }
      auto endBuildCluster = std::chrono::high_resolution_clock::now();
      auto durationBuildCluster = std::chrono::duration_cast<std::chrono::microseconds>(endBuildCluster - startBuildCluster);
      std::cout << "Time to build cls: " << durationBuildCluster.count() << std::endl;
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "Time for all sectors: " << duration.count() << std::endl;
    outFile << duration.count() << std::endl;
    // Fill Tree
    tClusters->Fill();
    clusters.clear();
  }
  // Write Tree to file
  fOut->Write();
  fOut->Close();
  outFile.close();
  return;
}

int main()
{
  findKrBoxCluster();
  return 0;
}