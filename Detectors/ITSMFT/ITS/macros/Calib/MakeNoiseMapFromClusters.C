#if !defined(__CLING__) || defined(__ROOTCLING__)
#include <iostream>
#include <vector>
#include <string>

#include <TFile.h>
#include <TTree.h>

#include "DataFormatsITSMFT/ClusterPattern.h"
#include "DataFormatsITSMFT/CompCluster.h"
#include "DataFormatsITSMFT/ROFRecord.h"
#include "DataFormatsITSMFT/NoiseMap.h"

#endif

void MakeNoiseMapFromClusters(std::string input = "o2clus_its.root", std::string output = "noise.root", int threshold = 3)
{
  TFile in(input.data());
  if (!in.IsOpen()) {
    std::cerr << "Can not open the input file " << input << '\n';
    return;
  }
  auto clusTree = (TTree*)in.Get("o2sim");
  if (!clusTree) {
    std::cerr << "Can not get cluster tree\n";
    return;
  }

  // Clusters
  std::vector<o2::itsmft::CompClusterExt>* clusters = nullptr;
  clusTree->SetBranchAddress("ITSClusterComp", &clusters);

  // Pixel patterns
  std::vector<unsigned char>* patternsPtr = nullptr;
  auto pattBranch = clusTree->GetBranch("ITSClusterPatt");
  if (pattBranch) {
    pattBranch->SetAddress(&patternsPtr);
  }

  //RO frames
  std::vector<o2::itsmft::ROFRecord>* rofVec = nullptr;
  clusTree->SetBranchAddress("ITSClustersROF", &rofVec);

  o2::itsmft::NoiseMap noiseMap(24120);

  int n1pix = 0;
  auto nevents = clusTree->GetEntries();
  for (int n = 0; n < nevents; n++) {
    clusTree->GetEntry(n);
    auto pattIt = patternsPtr->cbegin();
    for (const auto& rof : *rofVec) {
      auto clustersInFrame = rof.getROFData(*clusters);
      for (const auto& c : clustersInFrame) {
        if (c.getPatternID() != o2::itsmft::CompCluster::InvalidPatternID)
          continue;
        o2::itsmft::ClusterPattern patt(pattIt);
        if (patt.getRowSpan() != 1)
          continue;
        if (patt.getColumnSpan() != 1)
          continue;
        auto id = c.getSensorID();
        auto row = c.getRow();
        auto col = c.getCol();
        noiseMap.increaseNoiseCount(id, row, col);
        n1pix++;
      }
    }
  }

  int ncalib = noiseMap.dumpAboveThreshold(threshold);
  std::cout << "Threshold: " << threshold << "  number of pixels: " << ncalib << '\n';
  std::cout << "Number of 1-pixel clusters: " << n1pix << '\n';

  TFile out(output.data(), "new");
  if (!out.IsOpen()) {
    std::cerr << "Can not open the output file " << output << '\n';
    return;
  }
  out.WriteObject(&noiseMap, "Noise");
  out.Close();
}
