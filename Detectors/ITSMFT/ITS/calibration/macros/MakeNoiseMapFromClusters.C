#if !defined(__CLING__) || defined(__ROOTCLING__)
#include <iostream>
#include <vector>
#include <string>
#include <gsl/span>

#include <TFile.h>
#include <TTree.h>

#include "ITSCalibration/NoiseCalibrator.h"

#include "DataFormatsITSMFT/ClusterPattern.h"
#include "DataFormatsITSMFT/CompCluster.h"
#include "DataFormatsITSMFT/ROFRecord.h"
#include "CCDB/BasicCCDBManager.h"
#include "CCDB/CCDBTimeStampUtils.h"
#include "DataFormatsITSMFT/TopologyDictionary.h"
#endif

void MakeNoiseMapFromClusters(std::string input = "o2clus_its.root", bool only1pix = false, float probT = 3e-6, std::string output = "noise.root", long timestamp = 0)
{
  TFile out(output.data(), "new");
  if (!out.IsOpen()) {
    std::cerr << "The output file " << output << " already exists !";
    return;
  }

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

  o2::its::NoiseCalibrator calib(only1pix, probT);
  auto& mgr = o2::ccdb::BasicCCDBManager::instance();
  mgr.setURL("https://alice-ccdb.cern.ch");
  mgr.setTimestamp(timestamp ? timestamp : o2::ccdb::getCurrentTimestamp());

  calib.setClusterDictionary(mgr.get<o2::itsmft::TopologyDictionary>("ITS/Calib/ClusterDictionary"));

  auto nevents = clusTree->GetEntries();
  for (int n = 0; n < nevents; n++) {
    clusTree->GetEntry(n);
    calib.processTimeFrameClusters(*clusters, *patternsPtr, *rofVec);
  }
  calib.finalize();

  const auto& map = calib.getNoiseMap();
  out.WriteObject(&map, "ccdb_object");
  out.Close();
}
