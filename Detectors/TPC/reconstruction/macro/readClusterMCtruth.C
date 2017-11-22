// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file readCLusterMCtruth.cxx
/// \brief This macro demonstrates how to extract the MC truth information from
/// the clusters
/// \author Andi Mathis, TU München, andreas.mathis@ph.tum.de
#if !defined(__CLING__) || defined(__ROOTCLING__)
#include "TFile.h"
#include "TTree.h"

#include "TPCReconstruction/Cluster.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "SimulationDataFormat/MCCompLabel.h"

#include <vector>
#endif
void readClusterMCtruth(std::string filename)
{
  TFile *clusterFile = TFile::Open(filename.data());
  TTree *clusterTree = (TTree*)clusterFile->Get("o2sim");

  std::vector<o2::TPC::Cluster>* clusters = nullptr;
  clusterTree->SetBranchAddress("TPCClusterHW",&clusters);

  o2::dataformats::MCTruthContainer<o2::MCCompLabel> mMCTruthArray;
  o2::dataformats::MCTruthContainer<o2::MCCompLabel> *mcTruthArray(&mMCTruthArray);
  clusterTree->SetBranchAddress("TPCClusterHWMCTruth", &mcTruthArray);

  for(int iEvent=0; iEvent<clusterTree->GetEntriesFast(); ++iEvent) {
    int cluster = 0;
    clusterTree->GetEntry(iEvent);
    for(auto& inputcluster : *clusters) {
      gsl::span<const o2::MCCompLabel> mcArray = mMCTruthArray.getLabels(cluster);
      for(int j=0; j<static_cast<int>(mcArray.size()); ++j) {
        std::cout << "Cluster " << cluster << " from Event " <<
                     mMCTruthArray.getElement(mMCTruthArray.getMCTruthHeader(cluster).index+j).getEventID() << " with Track ID " <<
                     mMCTruthArray.getElement(mMCTruthArray.getMCTruthHeader(cluster).index+j).getTrackID() << "\n";
      }
      ++cluster;
    }
  }
}
