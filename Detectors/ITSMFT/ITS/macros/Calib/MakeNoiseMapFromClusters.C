#if !defined(__CLING__) || defined(__ROOTCLING__)
#include <iostream>
#include <vector>
#include <map>
#include <string>

#include <TFile.h>
#include <TTree.h>

#include "ITSBase/GeometryTGeo.h"
#include "DataFormatsITSMFT/CompCluster.h"
#include "DataFormatsITSMFT/TopologyDictionary.h"
#include "DataFormatsITSMFT/NoiseMap.h"

#endif

void MakeNoiseMapFromClusters(std::string dictfile = "ITSdictionary.bin", std::string input = "o2clus_its.root", std::string output = "noise.root")
{
  o2::itsmft::TopologyDictionary dict;
  std::ifstream file(dictfile.c_str());
  if (!file.good()) {
    std::cerr << "Cannot open the dictionary file: " << dictfile << '\n';
    return;
  }
  dict.readBinaryFile(dictfile);

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
  std::vector<o2::itsmft::CompClusterExt>* clusters = nullptr;
  clusTree->SetBranchAddress("ITSClusterComp", &clusters);

  o2::itsmft::NoiseMap noiseMap;

  auto nevents = clusTree->GetEntries();
  for (int n = 0; n < nevents; n++) {
    clusTree->GetEntry(n);

    for (const auto& c : *clusters) {
      auto pattID = c.getPatternID();
      auto npix = dict.getNpixels(pattID);

      if (npix > 1)
        continue;

      auto id = c.getSensorID();
      int row = c.getRow();
      int col = c.getCol();
      noiseMap.increaseNoiseCount(id, row, col);
    }
  }

  TFile out(output.data(), "new");
  if (!out.IsOpen()) {
    std::cerr << "Can not open the output file " << output << '\n';
    return;
  }
  out.WriteObject(&noiseMap, "Noise");
  out.Close();
}
