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

void MakeNoiseMapFromClusters(std::string input = "o2clus_its.root", std::string output = "noise.root", bool only1pix = true)
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

        auto id = c.getSensorID();
        auto row = c.getRow();
        auto col = c.getCol();
        auto colSpan = patt.getColumnSpan();
        auto rowSpan = patt.getRowSpan();

        if ((rowSpan == 1) && (colSpan == 1)) {
          noiseMap.increaseNoiseCount(id, row, col);
          continue;
        }

        if (only1pix)
          continue;

        auto nBits = rowSpan * colSpan;
        int ic = 0, ir = 0;
        for (unsigned int i = 2; i < patt.getUsedBytes() + 2; i++) {
          unsigned char tempChar = patt.getByte(i);
          int s = 128; // 0b10000000
          while (s > 0) {
            if ((tempChar & s) != 0) {
              noiseMap.increaseNoiseCount(id, row + ir, col + ic);
            }
            ic++;
            s >>= 1;
            if ((ir + 1) * ic == nBits) {
              break;
            }
            if (ic == colSpan) {
              ic = 0;
              ir++;
            }
          }
          if ((ir + 1) * ic == nBits) {
            break;
          }
        }
      }
    }
  }

  int ncalib = noiseMap.dumpAboveThreshold(3);
  std::cout << "Threshold: 3"
            << "  number of pixels: " << ncalib << '\n';

  TFile out(output.data(), "new");
  if (!out.IsOpen()) {
    std::cerr << "Can not open the output file " << output << '\n';
    return;
  }
  out.WriteObject(&noiseMap, "Noise");
  out.Close();
}
