/// \file CheckLookUp.C
/// Macro to check the correct identification of the cluster-topology ID. A dictionary of topologes (it can be generated with the macro CheckTopologies.C) is needed as input. The macro checks the correspondence between a topology and the identified entry in the dictionary. If the pattern is not the same, the input and the output are stored in a file.

#if !defined(__CLING__) || defined(__ROOTCLING__)
#include "TStopwatch.h"
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

#include "DataFormatsITSMFT/Cluster.h"
#include "DataFormatsITSMFT/ClusterTopology.h"
#include "ITSMFTReconstruction/BuildTopologyDictionary.h"
#include "ITSMFTReconstruction/LookUp.h"

#endif

bool verbose = false;

void CheckLookUp(std::string clusfile = "o2clus_its.root", std::string dictfile = "complete_dictionary.bin")
{

  using o2::itsmft::BuildTopologyDictionary;
  using o2::itsmft::Cluster;
  using o2::itsmft::ClusterTopology;
  using o2::itsmft::LookUp;
  using o2::itsmft::TopologyDictionary;

  LookUp finder("complete_dictionary.bin");
  TopologyDictionary dict;
  dict.ReadBinaryFile(dictfile.c_str());
  ofstream check_output("checkLU.txt");
  ofstream mist("mist.txt");
  TFile outroot("checkLU.root", "RECREATE");
  TH1F* hDistribution =
    new TH1F("hDistribution", ";TopologyID;frequency", 1060, -0.5, 1059.5);

  // Clusters
  TFile* file1 = TFile::Open(clusfile.data());
  TTree* clusTree = (TTree*)gFile->Get("o2sim");
  std::vector<Cluster>* clusArr = nullptr;
  clusTree->SetBranchAddress("ITSCluster", &clusArr);

  Int_t nevCl = clusTree->GetEntries(); // clusters in cont. readout may be
                                        // grouped as few events per entry
  int ievC = 0;
  int mistakes = 0;
  int total = 0;

  for (ievC = 0; ievC < nevCl; ievC++) {
    clusTree->GetEvent(ievC);
    Int_t nc = clusArr->size();
    printf("processing cluster event %d\n", ievC);
    bool restart = false;
    restart = (ievC == 0) ? true : false;
    while (nc--) {
      total++;
      // cluster is in tracking coordinates always
      Cluster& c = (*clusArr)[nc];
      int rowSpan = c.getPatternRowSpan();
      int columnSpan = c.getPatternColSpan();
      int nBytes = (rowSpan * columnSpan) >> 3;
      if (((rowSpan * columnSpan) % 8) != 0)
        nBytes++;
      unsigned char patt[Cluster::kMaxPatternBytes];
      c.getPattern(&patt[0], nBytes);
      ClusterTopology topology(rowSpan, columnSpan, patt);
      std::array<unsigned char, Cluster::kMaxPatternBytes + 2> pattExt =
        topology.getPattern(); // Get the pattern in extended format (the first two bytes are the number of rows/colums)
      if (verbose) {
        check_output << "input:" << endl
                     << endl;
        check_output << topology << endl;
        check_output << "output:" << endl
                     << endl;
      }
      int out_index = finder.findGroupID(rowSpan, columnSpan, patt); // Find ID in the dictionary
      std::array<unsigned char, Cluster::kMaxPatternBytes + 2> out_patt =
        dict.GetPattern(out_index).getPattern(); // Get the pattern corresponding to the ID
      hDistribution->Fill(out_index);
      if (verbose) {
        check_output << dict.GetPattern(out_index) << endl;
        check_output
          << "********************************************************"
          << endl;
      }
      for (int i = 0; i < Cluster::kMaxPatternBytes + 2; i++) {
        if (pattExt[i] != out_patt[i]) {
          mistakes++;
          mist << "input:" << endl
               << endl;
          mist << topology << endl;
          mist << "output:" << endl
               << endl;
          mist << dict.GetPattern(finder.findGroupID(rowSpan, columnSpan, patt))
               << endl;
          mist << "********************************************************"
               << endl;
          break;
        }
      }
    }
  }
  std::cout << "number of mismatch:" << mistakes << " / " << total << std::endl;
  hDistribution->Scale(1 / hDistribution->Integral());
  outroot.cd();
  hDistribution->Write();
}
