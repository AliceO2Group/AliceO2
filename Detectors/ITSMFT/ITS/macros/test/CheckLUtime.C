/// \file CheckLUtime.C
/// \brief Macro to measure the time necessaty for the identification of the topology IDs of the clusters generated in an event. A dictionary of topologies must be provided as input. An input file with the pattern for ALL the clusters must be provuded.

#if !defined(__CLING__) || defined(__ROOTCLING__)
#include <TAxis.h>
#include <TCanvas.h>
#include <TFile.h>
#include <TH1F.h>
#include <TH2F.h>
#include <TNtuple.h>
#include <TString.h>
#include <TStyle.h>
#include <TTree.h>
#include <fstream>
#include <string>
#include "TStopwatch.h"

#include "ITSMFTReconstruction/LookUp.h"
#include "DataFormatsITSMFT/ClusterPattern.h"
#include "DataFormatsITSMFT/CompCluster.h"
#include "DataFormatsITSMFT/ROFRecord.h"
#include "DetectorsCommonDataFormats/NameConf.h"

#endif

using namespace std;

void CheckLUtime(std::string clusfile = "o2clus_its.root", std::string dictfile = "")
{
  using o2::itsmft::ClusterPattern;
  using o2::itsmft::CompClusterExt;
  using o2::itsmft::CompCluster;
  using o2::itsmft::LookUp;
  using ROFRec = o2::itsmft::ROFRecord;

  TStopwatch sw;
  sw.Start();

  if (dictfile.empty()) {
    dictfile = o2::base::NameConf::getAlpideClusterDictionaryFileName(o2::detectors::DetID::ITS, "", ".bin");
  }
  std::ifstream file(dictfile.c_str());
  if (file.good()) {
    LOG(INFO) << "Running with dictionary: " << dictfile.c_str();
  } else {
    LOG(INFO) << "Running without dictionary !";
  }
  LookUp finder(dictfile.c_str());
  ofstream time_output("time.txt");

  ofstream realtime, cputime;
  realtime.open("realtime.txt", std::ofstream::out | std::ofstream::app);
  cputime.open("cputime.txt", std::ofstream::out | std::ofstream::app);

  // Clusters
  TFile* fileCl = TFile::Open(clusfile.data());
  TTree* clusTree = (TTree*)fileCl->Get("o2sim");
  std::vector<CompClusterExt>* clusArr = nullptr;
  clusTree->SetBranchAddress("ITSClusterComp", &clusArr);
  std::vector<unsigned char>* patternsPtr = nullptr;
  auto pattBranch = clusTree->GetBranch("ITSClusterPatt");
  if (pattBranch) {
    pattBranch->SetAddress(&patternsPtr);
  }

  // ROFrecords
  std::vector<ROFRec> rofRecVec, *rofRecVecP = &rofRecVec;
  clusTree->SetBranchAddress("ITSClustersROF", &rofRecVecP);
  clusTree->GetEntry(0);
  int nROFRec = (int)rofRecVec.size();

  auto pattIdx = patternsPtr->cbegin();

  int nClusters = 0;

  for (int irof = 0; irof < nROFRec; irof++) {
    const auto& rofRec = rofRecVec[irof];

    rofRec.print();

    for (int icl = 0; icl < rofRec.getNEntries(); icl++) {
      nClusters++;
      int clEntry = rofRec.getFirstEntry() + icl; // entry of icl-th cluster of this ROF in the vector of clusters
      // do we read MC data?

      const auto& cluster = (*clusArr)[clEntry];

      if (cluster.getPatternID() != CompCluster::InvalidPatternID) {
        LOG(WARNING) << "Clusters have already been generated with a dictionary! Quitting";
        return;
      }

      auto rowSpan = *pattIdx++;
      auto columnSpan = *pattIdx++;
      int nBytes = (rowSpan * columnSpan) >> 3;
      if (((rowSpan * columnSpan) % 8) != 0)
        nBytes++;
      unsigned char patt[ClusterPattern::MaxPatternBytes] = {0}, *p = &patt[0];
      while (nBytes--) {
        *p++ = *pattIdx++;
      }
      finder.findGroupID(rowSpan, columnSpan, patt);
    }
  }
  sw.Stop();
  realtime << sw.RealTime() / nClusters << std::endl;
  realtime.close();
  cputime << sw.CpuTime() / nClusters << std::endl;
  cputime.close();
  time_output << "Real time (s): " << sw.RealTime() / nClusters << "CPU time (s): " << sw.CpuTime() / nClusters << std::endl;
  std::cout << "Real time (s): " << sw.RealTime() / nClusters << " CPU time (s): " << sw.CpuTime() / nClusters << std::endl;
}
