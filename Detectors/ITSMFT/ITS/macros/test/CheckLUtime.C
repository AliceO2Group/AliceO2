/// \file CheckLUtime.C
/// \brief Macro to measure the time necessaty for the identification of the topology IDs of the clusters generated in an event. A dictionary of topologies must be provided as input.

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
#include "DataFormatsITSMFT/Cluster.h"
#include "DataFormatsITSMFT/CompCluster.h"
#include "DetectorsCommonDataFormats/NameConf.h"

#endif

using namespace std;

void CheckLUtime(std::string clusfile = "o2clus_its.root", std::string dictfile = "")
{
  using o2::itsmft::Cluster;
  using o2::itsmft::CompClusterExt;
  using o2::itsmft::LookUp;

  if (dictfile.empty()) {
    dictfile = o2::base::NameConf::getDictionaryFileName(o2::detectors::DetID::ITS, "", ".bin");
  }
  LookUp finder(dictfile.c_str());
  ofstream time_output("time.txt");

  ofstream realtime, cputime;
  realtime.open("realtime.txt", std::ofstream::out | std::ofstream::app);
  cputime.open("cputime.txt", std::ofstream::out | std::ofstream::app);

  TStopwatch timerLookUp;

  // Clusters
  TFile* file1 = TFile::Open(clusfile.data());
  TTree* clusTree = (TTree*)gFile->Get("o2sim");
  std::vector<CompClusterExt>* clusArr = nullptr;
  clusTree->SetBranchAddress("ITSClusterComp", &clusArr);
  std::vector<unsigned char>* patternsPtr = nullptr;
  auto pattBranch = clusTree->GetBranch("ITSClusterPatt");
  if (pattBranch) {
    pattBranch->SetAddress(&patternsPtr);
  }

  Int_t nevCl = clusTree->GetEntries(); // clusters in cont. readout may be grouped as few events per entry
  int ievC = 0, ievH = 0;

  for (ievC = 0; ievC < nevCl; ievC++) {
    clusTree->GetEvent(ievC);
    Int_t nc = clusArr->size();
    printf("processing cluster event %d\n", ievC);
    bool restart = false;
    restart = (ievC == 0) ? true : false;
    timerLookUp.Start(restart);
    auto pattIdx = patternsPtr->cbegin();
    for (int i = 0; i < nc; i++) {
      CompClusterExt& c = (*clusArr)[i];
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
    timerLookUp.Stop();
  }
  realtime << timerLookUp.RealTime() / nevCl << std::endl;
  realtime.close();
  cputime << timerLookUp.CpuTime() / nevCl << std::endl;
  cputime.close();
  time_output << "Real time (s): " << timerLookUp.RealTime() / nevCl << "CPU time (s): " << timerLookUp.CpuTime() / nevCl << std::endl;
  std::cout << "Real time (s): " << timerLookUp.RealTime() / nevCl << " CPU time (s): " << timerLookUp.CpuTime() / nevCl << std::endl;
}
