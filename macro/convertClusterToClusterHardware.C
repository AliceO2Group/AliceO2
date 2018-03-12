// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#if !defined(__CLING__) || defined(__ROOTCLING__)
#include <vector>
#include <fstream>
#include <iostream>
#include "TSystem.h"

#include "TROOT.h"
#include "TFile.h"
#include "TString.h"
#include "TChain.h"
#include "TTree.h"

#include "SimulationDataFormat/MCTruthContainer.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "DataFormatsTPC/ClusterHardware.h"
#include "DataFormatsTPC/Helpers.h"
#include "DataFormatsTPC/Cluster.h"
#include "DataFormatsTPC/Constants.h"

#include "TPCBase/CRU.h"
#else
#pragma cling load("libTPCReconstruction")
#pragma cling load("libDataFormatsTPC")
#endif

using namespace o2;
using namespace o2::TPC;
using namespace o2::dataformats;
using namespace std;

using MCLabelContainer = MCTruthContainer<MCCompLabel>;

int convertClusterToClusterHardware(TString infile = "", TString outfile = "")
{
  if (infile.EqualTo("") || outfile.EqualTo("")) {
    printf("Filename missing\n");
    return (1);
  }
  ClusterHardwareContainer8kb clusterContainerMemory;
  int maxClustersPerContainer = clusterContainerMemory.getMaxNumberOfClusters();
  ClusterHardwareContainer& clusterContainer = *clusterContainerMemory.getContainer();
  MCLabelContainer outMCLabels;

  TChain c("o2sim");
  c.AddFile(infile);
  std::vector<o2::TPC::Cluster>* inClusters = nullptr;
  c.SetBranchAddress("TPCClusterHW", &inClusters);
  MCLabelContainer* inMCLabels = nullptr;
  c.SetBranchAddress("TPCClusterHWMCTruth", &inMCLabels);

  TFile fout(outfile, "recreate");
  TTree tout("clustersHardware", "clustersHardware");
  tout.Branch("clusters", &clusterContainerMemory);
  tout.Branch("clustersMCTruth", &outMCLabels);

  int nClusters = 0, nContainers = 0, nMCLabels = 0;

  const int nentries = c.GetEntries();
  for (int iEvent = 0; iEvent < nentries; iEvent++) {
    c.GetEntry(iEvent);
    if (!inClusters->size())
      continue;

    unsigned int iCurrentCluster = 0;
    while (iCurrentCluster < inClusters->size()) {
      clusterContainer.CRU = (*inClusters)[iCurrentCluster].getCRU();
      clusterContainer.numberOfClusters = 0;
      clusterContainer.timeBinOffset = 0xFFFFFFFF;
      for (unsigned int icluster = iCurrentCluster;
           icluster < inClusters->size() && clusterContainer.numberOfClusters < maxClustersPerContainer &&
           (*inClusters)[icluster].getCRU() == clusterContainer.CRU;
           icluster++) {
        clusterContainer.numberOfClusters++;
        if ((*inClusters)[icluster].getTimeMean() < clusterContainer.timeBinOffset)
          clusterContainer.timeBinOffset = (*inClusters)[icluster].getTimeMean();
      }

      outMCLabels.clear();
      for (unsigned int icluster = 0; icluster < clusterContainer.numberOfClusters; icluster++) {
        const auto& cluster = (*inClusters)[iCurrentCluster + icluster];
        ClusterHardware& oCluster = clusterContainer.clusters[icluster];
        oCluster.setCluster(cluster.getPadMean(), cluster.getTimeMean() - clusterContainer.timeBinOffset,
                            cluster.getPadSigma() * cluster.getPadSigma(),
                            cluster.getTimeSigma() * cluster.getTimeSigma(), cluster.getQmax(), cluster.getQ(),
                            cluster.getRow(), 0);
        for (const auto& element : inMCLabels->getLabels(iCurrentCluster + icluster)) {
          outMCLabels.addElement(icluster, element);
          nMCLabels++;
        }
      }
      iCurrentCluster += clusterContainer.numberOfClusters;
      nClusters += clusterContainer.numberOfClusters;
      nContainers++;
      tout.Fill();
    }
  }
  tout.Write();
  fout.Close();

  printf("Wrote %d clusters, %d containers, %d MC labels\n", nClusters, nContainers, nMCLabels);
  return (0);
}
