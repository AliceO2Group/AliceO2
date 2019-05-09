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

#include "ITSMFTReconstruction/BuildTopologyDictionary.h"
#include "ITSMFTReconstruction/LookUp.h"
#include "DataFormatsITSMFT/Cluster.h"
#include "DataFormatsITSMFT/ClusterTopology.h"

#endif

void CheckLUtime(std::string clusfile = "o2clus_its.root", std::string dictfile = "complete_dictionary.bin")
{

  using o2::itsmft::BuildTopologyDictionary;
  using o2::itsmft::Cluster;
  using o2::itsmft::ClusterTopology;
  using o2::itsmft::LookUp;
  using o2::itsmft::TopologyDictionary;

  LookUp finder(dictfile.c_str());
  TopologyDictionary dict;
  ofstream time_output("time.txt");

  ofstream realtime, cputime;
  realtime.open("realtime.txt", std::ofstream::out | std::ofstream::app);
  cputime.open("cputime.txt", std::ofstream::out | std::ofstream::app);

  TStopwatch timerLookUp;

  // Clusters
  TFile* file1 = TFile::Open(clusfile.data());
  TTree* clusTree = (TTree*)gFile->Get("o2sim");
  std::vector<Cluster>* clusArr = nullptr;
  clusTree->SetBranchAddress("ITSCluster", &clusArr);

  Int_t nevCl = clusTree->GetEntries(); // clusters in cont. readout may be grouped as few events per entry
  int ievC = 0, ievH = 0;

  for (ievC = 0; ievC < nevCl; ievC++) {
    clusTree->GetEvent(ievC);
    Int_t nc = clusArr->size();
    printf("processing cluster event %d\n", ievC);
    bool restart = false;
    restart = (ievC == 0) ? true : false;
    timerLookUp.Start(restart);
    while (nc--) {
      // cluster is in tracking coordinates always
      Cluster& c = (*clusArr)[nc];
      int rowSpan = c.getPatternRowSpan();
      int columnSpan = c.getPatternColSpan();
      int nBytes = (rowSpan * columnSpan) >> 3;
      if (((rowSpan * columnSpan) % 8) != 0)
        nBytes++;
      unsigned char patt[Cluster::kMaxPatternBytes];
      c.getPattern(&patt[0], nBytes);
      finder.findGroupID(rowSpan, columnSpan, patt);
    }
    timerLookUp.Stop();
  }
  realtime << timerLookUp.RealTime() / nevCl << std::endl;
  realtime.close();
  cputime << timerLookUp.CpuTime() / nevCl << std::endl;
  cputime.close();
  time_output << "Real time (s): " << timerLookUp.RealTime() / nevCl << "CPU time (s): " << timerLookUp.CpuTime() / nevCl << endl;
  cout << "Real time (s): " << timerLookUp.RealTime() / nevCl << " CPU time (s): " << timerLookUp.CpuTime() / nevCl << endl;
}
