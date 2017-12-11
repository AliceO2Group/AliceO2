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

#include "SimulationDataFormat/MCTruthContainer.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "DataFormatsTPC/ClusterHardware.h"
#include "DataFormatsTPC/Helpers.h"
#include "TPCReconstruction/Cluster.h"
#include "TPCBase/Constants.h"
#include "TPCBase/CRU.h"
#endif

using namespace o2::TPC;
using namespace o2::DataFormat::TPC;
using namespace o2;
using namespace o2::dataformats;
using namespace std;

using MCLabelContainer = MCTruthContainer<MCCompLabel>;

void convertClusterToClusterHardware(TString infile = "o2clus.root", TString outfile = "clusterHardware.root") {
  gSystem->Load("libTPCReconstruction.so");
  
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
  TTree tout("clustersHardware","clustersHardware");
  tout.Branch("clusters", &clusterContainerMemory);
  tout.Branch("clustersMCTruth", &outMCLabels);

  int nClusters = 0, nContainers = 0, nMCLabels = 0;

  const int nentries = c.GetEntries();
  for (int iEvent=0;iEvent < nentries;iEvent++) {
    c.GetEntry(iEvent);
    if (!inClusters->size()) continue;

    unsigned int iCurrentCluster = 0;
    while (iCurrentCluster < inClusters->size())
    {
      clusterContainer.mCRU = (*inClusters)[iCurrentCluster].getCRU();
      clusterContainer.mNumberOfClusters = 0;
      clusterContainer.mTimeBinOffset = 0xFFFFFFFF;
      for (unsigned int icluster = iCurrentCluster;icluster < inClusters->size() && clusterContainer.mNumberOfClusters < maxClustersPerContainer && (*inClusters)[icluster].getCRU() == clusterContainer.mCRU;icluster++)
      {
        clusterContainer.mNumberOfClusters++;
        if ((*inClusters)[icluster].getTimeMean() < clusterContainer.mTimeBinOffset) clusterContainer.mTimeBinOffset = (*inClusters)[icluster].getTimeMean();
      }
      
      outMCLabels.clear();
      for (unsigned int icluster = 0;icluster < clusterContainer.mNumberOfClusters;icluster++) {
        const auto& cluster = (*inClusters)[iCurrentCluster + icluster];
        
        float mPadPre;                //< Quantity needed to compute the pad
        float mTimePre;               //< Quantity needed to compute the time
        float mSigmaPad2Pre;          //< Quantity needed to compute the sigma^2 of the pad
        float mSigmaTime2Pre;         //< Quantity needed to compute the sigma^2 of the time
        uint16_t mQMax;               //< QMax of the cluster
        uint16_t mQTot;               //< Total charge of the cluster
        uint8_t mRow;                 //< Row of the cluster (local, needs to add PadRegionInfo::getGlobalRowOffset
        uint8_t mFlags;               //< Flags of the cluster
        ClusterHardware& oCluster = clusterContainer.mClusters[icluster];
        oCluster.mQMax = cluster.getQmax() + 0.5;
        oCluster.mQTot = cluster.getQ() + 0.5;
        oCluster.mPadPre = cluster.getPadMean() * oCluster.mQTot;
        oCluster.mTimePre = (cluster.getTimeMean() - clusterContainer.mTimeBinOffset) * oCluster.mQTot;
        oCluster.mSigmaPad2Pre = cluster.getPadSigma() * cluster.getPadSigma() * oCluster.mQTot * oCluster.mQTot + oCluster.mPadPre * oCluster.mPadPre;
        oCluster.mSigmaTime2Pre = cluster.getTimeSigma() * cluster.getTimeSigma() * oCluster.mQTot * oCluster.mQTot + oCluster.mTimePre * oCluster.mTimePre;
        oCluster.mRow = cluster.getRow();
        oCluster.mFlags = 0;
        gsl::span<const MCCompLabel> mcArray = inMCLabels->getLabels(iCurrentCluster + icluster);
        for (int k = 0;k < mcArray.size();k++) {
          outMCLabels.addElement(icluster, mcArray[k]);
          nMCLabels++;
        }
      }
      iCurrentCluster += clusterContainer.mNumberOfClusters;
      nClusters += clusterContainer.mNumberOfClusters;
      nContainers++;
      tout.Fill();
    }
  }
  tout.Write();
  fout.Close();
  
  printf("Wrote %d clusters, %d containers, %d MC labels\n", nClusters, nContainers, nMCLabels);
}
