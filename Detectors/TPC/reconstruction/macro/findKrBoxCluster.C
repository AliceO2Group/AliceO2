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

void findKrBoxCluster()
{
  // Read the digits:
  TFile* file = new TFile("tpcdigits.root");
  TTree* tree = (TTree*)file->Get("o2sim");
  Long64_t nEntries = tree->GetEntries();
  std::cout << "The Tree has " << nEntries << " Entries." << std::endl;

  // Initialize File for later writing
  TFile* f = new TFile("boxClustersSectors.root", "RECREATE", "Clusters");
  TTree* T = new TTree("T", "Clusters");

  // Tree will be filled with a vector of clusters
  std::vector<o2::tpc::KrCluster> vCluster{};
  T->Branch("cluster", &vCluster);

  std::array<std::vector<o2::tpc::Digit>*, 36> DigitizedSignal;
  for (int iSec = 0; iSec < DigitizedSignal.size(); ++iSec) {
    DigitizedSignal[iSec] = nullptr;
    tree->SetBranchAddress(Form("TPCDigit_%d", iSec), &DigitizedSignal[iSec]);
  }

  // Now everything can get processed
  // Loop over all events
  for (int iEvent = 0; iEvent < nEntries; ++iEvent) {
    std::cout << iEvent + 1 << "/" << nEntries << std::endl;
    tree->GetEntry(iEvent);
    // Each event consists of sectors (atm only two)
    for (int i = 0; i < 36; i++) {
      auto sector = DigitizedSignal[i];
      if (sector->size() != 0) {
        // Create ClusterFinder Object on Heap since creation on stack fails
        // Probably due to too much memory consumption
        o2::tpc::KrBoxClusterFinder* cluster = new o2::tpc::KrBoxClusterFinder(*sector);
        std::vector<std::tuple<int, int, int>> localMaxima = cluster->findLocalMaxima();
        // Loop over cluster centers
        for (const std::tuple<int, int, int>& coords : localMaxima) {
          int padMax = std::get<0>(coords);
          int rowMax = std::get<1>(coords);
          int timeMax = std::get<2>(coords);
          // Build total cluster
          o2::tpc::KrCluster tempCluster = cluster->buildCluster(padMax, rowMax, timeMax);
          tempCluster.sector = i;
          vCluster.emplace_back(tempCluster);
        }
        // Clean up memory:
        delete cluster;
        cluster = nullptr;
      }
    }
    // Fill Tree
    T->Fill();
    vCluster.clear();
  }
  // Write Tree to file
  f->cd();
  T->Write();
  f->Close();
  return;
}
