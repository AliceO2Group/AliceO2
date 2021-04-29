#if !defined(__CLING__) || defined(__ROOTCLING__)
#include <TCanvas.h>
#include <TClonesArray.h>
#include <TFile.h>
#include <TH1F.h>
#include <TTreeReader.h>
#include <TTreeReaderValue.h>

#include "DataFormatsITSMFT/CompCluster.h"
#include "DataFormatsITSMFT/TopologyDictionary.h"
#include "DetectorsCommonDataFormats/NameConf.h"

#include <string>
#include <vector>
#endif

// Compare the topology distribution from the cluster finder with that in the dictionary

using namespace std;

void compareTopologyDistributions(
  string cluster_file_name = "o2clus_its.root",
  string dictionary_file_name = "",
  string output_file_name = "comparison.root")
{
  if (dictionary_file_name.empty()) {
    dictionary_file_name = o2::base::NameConf::getAlpideClusterDictionaryFileName(o2::detectors::DetID::ITS, "", ".bin");
  }
  o2::itsmft::TopologyDictionary dict;
  dict.readBinaryFile(dictionary_file_name.c_str());
  int dict_size = dict.getSize();
  TH1F* hDictio = nullptr;
  o2::itsmft::TopologyDictionary::getTopologyDistribution(dict, hDictio, "hComplete");
  hDictio->SetDirectory(0);

  TFile* cluster_file = TFile::Open(cluster_file_name.c_str());
  TTreeReader reader("o2sim", cluster_file);
  TTreeReaderValue<std::vector<o2::itsmft::CompClusterExt>> comp_clus_vec(
    reader, "ITSClusterComp");
  TH1F* hRec =
    new TH1F("hRec", ";Topology ID;", dict_size, -0.5, dict_size - 0.5);
  while (reader.Next()) {
    for (auto& comp_clus : *comp_clus_vec) {
      int clusID = comp_clus.getPatternID();
      hRec->Fill(clusID);
    }
  }
  hRec->Scale(1 / hRec->Integral());

  TFile output_file(output_file_name.c_str(), "RECREATE");
  hRec->Write();
  hDictio->Write();
  TCanvas* cOut = new TCanvas("cOut", "cOut", 800, 600);
  cOut->SetLogy();
  hDictio->GetYaxis()->SetRangeUser(1.e-8, 1.2);
  hDictio->Draw("histo");
  hRec->SetMarkerStyle(20);
  hRec->SetMarkerSize(0.5);
  hRec->Draw("PSAME");
  cOut->Write();
}
