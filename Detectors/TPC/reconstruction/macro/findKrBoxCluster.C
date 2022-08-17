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

/// \file findKrBoxCluster.C
/// \brief This macro retrieves clusters from Krypton and X-Ray runs, input tpcdigits.root
/// \author Philip Hauer <philip.hauer@cern.ch>

//#if !defined(__CLING__) || defined(__ROOTCLING__)
#include "TCanvas.h"
#include "TFile.h"
#include "TChain.h"
#include "TGrid.h"

#include "DataFormatsTPC/KrCluster.h"
#include "TPCReconstruction/KrBoxClusterFinder.h"
#include "DataFormatsTPC/Digit.h"
#include "TPCBase/Utils.h"

#include <array>
#include <iostream>
#include <tuple>
#include <vector>
//#endif

void findKrBoxCluster(std::string inputFile = "tpcdigits.root", std::string outputFile = "BoxClusters.root", std::string_view gainMapFile = "")
{
  int lastTimeBin = 1000;
  ULong_t runRead = -1;
  uint32_t run = 0;
  uint32_t firstOrbit = 0;
  uint32_t tfCounter = 0;

  // Read the digits:
  TChain* tree = o2::tpc::utils::buildChain(fmt::format("cat {}", inputFile), "o2sim", "o2sim");
  Long64_t nEntries = tree->GetEntries();
  std::cout << "The Tree has " << nEntries << " Entries." << std::endl;

  // Initialize File for later writing
  TFile* fOut = new TFile(outputFile.c_str(), "RECREATE");
  TTree* tClusters = new TTree("Clusters", "Clusters");

  // Create KrBoxClusterFinder object, memory is only allocated once
  auto clFinder = std::make_unique<o2::tpc::KrBoxClusterFinder>();
  auto& clusters = clFinder->getClusters();
  // Create a Branch for each sector:
  tClusters->Branch("cls", &clusters);
  tClusters->Branch("run", &run);
  tClusters->Branch("firstOrbit", &firstOrbit);
  tClusters->Branch("tfCounter", &tfCounter);

  std::array<std::vector<o2::tpc::Digit>*, 36> digitizedSignal;
  for (size_t iSec = 0; iSec < digitizedSignal.size(); ++iSec) {
    digitizedSignal[iSec] = nullptr;
    tree->SetBranchAddress(Form("TPCDigit_%zu", iSec), &digitizedSignal[iSec]);
  }
  tree->SetBranchAddress("run", &runRead);
  tree->SetBranchAddress("firstOrbit", &firstOrbit);
  tree->SetBranchAddress("tfCounter", &tfCounter);

  if (gainMapFile.size()) {
    clFinder->loadGainMapFromFile(gainMapFile);
  }

  // clFinder->setMinNumberOfNeighbours(0);
  // clFinder->setMinQTreshold(0);
  clFinder->setMaxTimes(lastTimeBin);

  // Now everything can get processed
  // Loop over all events
  for (int iEvent = 0; iEvent < nEntries; ++iEvent) {
    std::cout << iEvent + 1 << "/" << nEntries << std::endl;
    tree->GetEntry(iEvent);
    run = uint32_t(runRead);

    for (int i = 0; i < 36; i++) {
      auto sector = digitizedSignal[i];
      if (sector->size() == 0) {
        continue;
      }
      auto& digits = *sector;
      std::sort(digits.begin(), digits.end(), [](const auto& a, const auto& b) {
        if (a.getTimeStamp() < b.getTimeStamp()) {
          return true;
        }
        if (a.getTimeStamp() == b.getTimeStamp()) {
          if (a.getRow() < b.getRow()) {
            return true;
          } else if (a.getRow() == b.getRow()) {
            return a.getPad() < b.getPad();
          }
        }
        return false;
      });

      // std::cout << "Processing sector " << i << "\n";

      clFinder->loopOverSector(*sector, i);
    }
    // Fill Tree
    tClusters->Fill();
    clusters.clear();
  }
  // Write Tree to file
  fOut->Write();
  fOut->Close();
  return;
}

int main()
{
  findKrBoxCluster();
  return 0;
}
