#if !defined(__CLING__) || defined(__ROOTCLING__)
#include <TCanvas.h>
#include <TClonesArray.h>
#include <TFile.h>
#include <TH1F.h>
#include <TTreeReader.h>
#include <TTreeReaderValue.h>

#include "DataFormatsITSMFT/Cluster.h"
#include "DataFormatsITSMFT/CompCluster.h"

#include <string>
#include <vector>
#endif

// Compare the topology distribution from the cluster finder with that in the dictionary

void compareTopologyDistributions(
  string cluster_file_name = "o2clus_its.root",
  string dictionary_file_name = "histograms.root",
  string output_file_name = "comparison.root")
{

  TFile dictionary_file(dictionary_file_name.c_str());
  TH1F* hDictio = (TH1F*)dictionary_file.Get("hComplete");
  hDictio->SetDirectory(0);
  int nBins = hDictio->GetNbinsX();
  float lower_bound = hDictio->GetXaxis()->GetBinLowEdge(1);
  float upper_bound = hDictio->GetXaxis()->GetBinLowEdge(nBins + 1);

  TFile* cluster_file = TFile::Open(cluster_file_name.c_str());
  TTreeReader reader("o2sim", cluster_file);
  TTreeReaderValue<std::vector<o2::itsmft::CompClusterExt>> comp_clus_vec(
    reader, "ITSClusterComp");
  TH1F* hRec =
    new TH1F("hRec", ";Topology ID;", nBins, lower_bound, upper_bound);
  while (reader.Next()) {
    for (auto& comp_clus : *comp_clus_vec) {
      int clusID = comp_clus.getPatternID();
      hRec->Fill(clusID);
    }
  }
  hRec->Scale(1 / hRec->Integral());

  TFile output_file(output_file_name.c_str(), "RECREATE");
  TCanvas* cOut = new TCanvas("cOut", "cOut");
  cOut->DrawFrame(-0.5, 1e-8, 200, 1, ";Topology ID;");
  cOut->SetLogy();
  hDictio->Draw("histo");
  hRec->SetMarkerStyle(20);
  hRec->SetMarkerSize(0.5);
  hRec->Draw("PSAME");
  cOut->Write();
}
