/// \file CheckLookUp.C
/// Macro to check the correct identification of the cluster-topology ID. A dictionary of topologes (it can be generated with the macro CheckTopologies.C) is needed as input. The macro checks the correspondence between a topology and the identified entry in the dictionary. If the pattern is not the same, the input and the output are stored in a file.

#if !defined(__CLING__) || defined(__ROOTCLING__)
#include "TStopwatch.h"
#include <TAxis.h>
#include <TCanvas.h>
#include <TLegend.h>
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
#include "DataFormatsITSMFT/CompCluster.h"
#include "ITSMFTReconstruction/LookUp.h"

#endif

bool verbose = true;

void CheckLookUp(std::string clusfile = "o2clus_its_comp.root", std::string dictfile = "complete_dictionary.bin")
{
#ifndef _ClusterTopology_
  std::cout << "This macro needs clusters in full format!" << std::endl;
  return;
#else

  // This macro needs itsmft::Clusters to be compiled with topology information
  using o2::itsmft::BuildTopologyDictionary;
  using o2::itsmft::Cluster;
  using o2::itsmft::ClusterTopology;
  using o2::itsmft::LookUp;
  using o2::itsmft::TopologyDictionary;

  LookUp finder("complete_dictionary.bin");
  TopologyDictionary dict;
  dict.readBinaryFile(dictfile.c_str());
  ofstream check_output("checkLU.txt");
  ofstream mist("mist.txt");
  TFile outroot("checkLU.root", "RECREATE");
  TH1F* hDistribution =
    new TH1F("hDistribution", ";TopologyID;frequency", dict.getSize(), -0.5, dict.getSize() - 0.5);

  // Clusters
  TFile* file1 = TFile::Open(clusfile.data());
  TTree* clusTree = (TTree*)gFile->Get("o2sim");
  std::vector<Cluster>* clusArr = nullptr;
  clusTree->SetBranchAddress("ITSCluster", &clusArr);

  Int_t nevCl = clusTree->GetEntries(); // clusters in cont. readout may be
                                        // grouped as few events per entry
  int ievC = 0;
  int mistakes = 0;
  int in_groups = 0;
  int mistakes_outside_groups = 0;
  int total = 0;

  for (ievC = 0; ievC < nevCl; ievC++) {
    clusTree->GetEvent(ievC);
    Int_t nc = clusArr->size();
    printf("processing cluster event %d\n", ievC);

    auto pattIdx = patternsPtr->cbegin();
    for (int i = 0; i < nc; i++) {
      total++;
      // cluster is in tracking coordinates always
      Cluster& c = (*clusArr)[i];
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
      bool bGroup = finder.isGroup(out_index);
      if (bGroup)
        in_groups++;
      std::array<unsigned char, Cluster::kMaxPatternBytes + 2> out_patt =
        dict.getPattern(out_index).getPattern(); // Get the pattern corresponding to the ID
      hDistribution->Fill(out_index);
      if (verbose) {
        check_output << dict.getPattern(out_index) << endl;
        check_output
          << "********************************************************"
          << endl;
      }
      for (int i = 0; i < Cluster::kMaxPatternBytes + 2; i++) {
        if (pattExt[i] != out_patt[i]) {
          mistakes++;
          if (!bGroup)
            mistakes_outside_groups++;
          mist << "input:" << endl
               << endl;
          mist << topology << endl;
          mist << "output:" << endl
               << "isGroup: " << std::boolalpha << dict.isGroup(finder.findGroupID(rowSpan, columnSpan, patt)) << endl
               << endl;
          mist << dict.getPattern(finder.findGroupID(rowSpan, columnSpan, patt))
               << endl;
          mist << "********************************************************"
               << endl;
          break;
        }
      }
    }
  }
  std::cout << "number of mismatch: " << mistakes << " / " << total << std::endl;
  std::cout << "number of clusters in groups: " << in_groups << " / " << total << std::endl;
  std::cout << "number of mismatch putside roups: " << mistakes_outside_groups << " / " << total << std::endl;
  hDistribution->Scale(1 / hDistribution->Integral());
  hDistribution->SetMarkerColor(kBlack);
  hDistribution->SetLineColor(kBlack);
  hDistribution->SetMarkerStyle(20);
  hDistribution->SetMarkerSize(0.5);
  outroot.cd();
  TCanvas* cv = new TCanvas("cv", "check_distribution", 800, 600);
  TH1F* hDict = nullptr;
  o2::itsmft::TopologyDictionary::getTopologyDistribution(dict, hDict, "hDictionary");
  hDict->SetDirectory(0);
  cv->cd();
  cv->SetLogy();
  hDict->GetYaxis()->SetRangeUser(1e-6, 1.2);
  hDict->Draw("histo");
  hDistribution->Draw("PE SAME");
  hDistribution->Write();
  TLegend* leg = new TLegend(0.65, 0.72, 0.89, 0.86, "", "brNDC");
  leg->SetBorderSize(0);
  leg->SetTextSize(0.027);
  leg->AddEntry(hDict, "Dictionary", "F");
  leg->AddEntry(hDistribution, "Topology distribution", "PE");
  leg->Draw();
  cv->Write();
#endif
}
