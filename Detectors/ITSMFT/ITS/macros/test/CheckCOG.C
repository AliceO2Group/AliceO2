/// \file CheckCOG.C
/// Macros to test the generation of a dictionary of topologies. Three dictionaries are generated: one with signal-cluster only, one with noise-clusters only and one with all the clusters.

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

#include "MathUtils/Utils.h"
#include "ITSBase/GeometryTGeo.h"
#include "DataFormatsITSMFT/Cluster.h"
#include "DataFormatsITSMFT/CompCluster.h"
#include "DataFormatsITSMFT/TopologyDictionary.h"
#include "MathUtils/Cartesian3D.h"

#endif

void CheckCOG(std::string clusfile = "o2clus_its.root", std::string inputGeom = "O2geometry.root", std::string dictionary_file = "complete_dictionary.bin")
{
  gStyle->SetOptStat(0);
  using namespace o2::base;
  using namespace o2::its;

  using o2::itsmft::Cluster;
  using o2::itsmft::CompClusterExt;
  using o2::itsmft::TopologyDictionary;

  // Geometry
  o2::base::GeometryManager::loadGeometry(inputGeom, "FAIRGeom");
  auto gman = o2::its::GeometryTGeo::Instance();
  gman->fillMatrixCache(o2::utils::bit2Mask(o2::TransformType::T2L, o2::TransformType::T2GRot,
                                            o2::TransformType::L2G)); // request cached transforms

  // Clusters
  TFile* file1 = TFile::Open(clusfile.data());
  TTree* clusTree = (TTree*)gFile->Get("o2sim");
  std::vector<Cluster>* clusArr = nullptr;
  clusTree->SetBranchAddress("ITSCluster", &clusArr);
  std::vector<CompClusterExt>* compclusArr = nullptr;
  clusTree->SetBranchAddress("ITSClusterComp", &compclusArr);

  int nevCl = clusTree->GetEntries(); // clusters in cont. readout may be grouped as few events per entry
  int ievC = 0;
  int lastReadHitEv = -1;

  TopologyDictionary dict(dictionary_file.c_str());

  TH1F* hTotalX = new TH1F("hTotalX", "All entries;x_{full} - x_{compact} (#mum);counts", 81, -40.5, 40.5);
  TH1F* hTotalZ = new TH1F("hTotalZ", "All entries;z_{full} - z_{compact} (#mum);counts", 81, -40.5, 40.5);
  TH1F* hCommonX = new TH1F("hCommonX", "Common topologies;x_{full} - x_{compact} (#mum);counts", 81, -40.5, 40.5);
  TH1F* hCommonZ = new TH1F("hCommonZ", "Common topologies;z_{full} - z_{compact} (#mum);counts", 81, -40.5, 40.5);
  TH1F* hGroupX = new TH1F("hGroupX", "Groups;x_{full} - x_{compact} (#mum);counts", 81, -40.5, 40.5);
  TH1F* hGroupZ = new TH1F("hGroupZ", "Groups;z_{full} - z_{compact} (#mum);counts", 81, -40.5, 40.5);
  std::ofstream fOut("COGdebug.txt");
  TCanvas* cTotal = new TCanvas("cTotal", "All topologies");
  cTotal->Divide(2, 1);
  TCanvas* cCommon = new TCanvas("cCommon", "Common topologies");
  cCommon->Divide(2, 1);
  TCanvas* cGroup = new TCanvas("cGroup", "Gourps of rare topologies");
  cGroup->Divide(2, 1);
  for (ievC = 0; ievC < nevCl; ievC++) {
    clusTree->GetEvent(ievC);
    int nc = clusArr->size();
    int nc_comp = compclusArr->size();
    if (nc != nc_comp) {
      std::cout << "The branches has different entries" << std::endl;
    }
    printf("processing cluster event %d\n", ievC);

    while (nc--) {
      // cluster is in tracking coordinates always
      Cluster& c = (*clusArr)[nc];
      const auto locC = c.getXYZLoc(*gman); // convert from tracking to local frame

      CompClusterExt& cComp = (*compclusArr)[nc];
      Point3D<float> locComp = dict.getClusterCoordinates(cComp);

      float xComp = locComp.X();
      float zComp = locComp.Z();

      float dx = (locC.X() - xComp) * 10000;
      float dz = (locC.Z() - zComp) * 10000;

      hTotalX->Fill(dx);
      hTotalZ->Fill(dx);

      fOut << Form("groupID: %d\n", cComp.getPatternID());
      fOut << Form("is group: %o\n", dict.IsGroup(cComp.getPatternID()));
      fOut << Form("x_full: %.4f x_comp: %.4f dx: %.4f x_shift: %.4f\n", locC.X(), xComp, dx / 10000, dict.GetXcog(cComp.getPatternID()));
      fOut << Form("z_full: %.4f z_comp: %.4f dZ: %.4f z_shift: %.4f\n", locC.Z(), zComp, dz / 10000, dict.GetZcog(cComp.getPatternID()));
      fOut << dict.GetPattern(cComp.getPatternID());
      fOut << Form("***************************************\n");

      if (dict.IsGroup(cComp.getPatternID())) {
        hGroupX->Fill(dx);
        hGroupZ->Fill(dx);
      } else {
        hCommonX->Fill(dx);
        hCommonZ->Fill(dx);
      }
    }
  }
  TFile output_file("COG_diff.root", "recreate");
  hTotalX->Write();
  hTotalZ->Write();
  hGroupX->Write();
  hGroupZ->Write();
  hCommonX->Write();
  hCommonZ->Write();
  cTotal->cd(1);
  gPad->SetLogy();
  hTotalX->Draw();
  cTotal->cd(2);
  gPad->SetLogy();
  hTotalZ->Draw();
  cTotal->Print("cTotal.pdf");
  cCommon->cd(1);
  gPad->SetLogy();
  hCommonX->Draw();
  cCommon->cd(2);
  gPad->SetLogy();
  hCommonZ->Draw();
  cCommon->Print("cCommon.pdf");
  cGroup->cd(1);
  gPad->SetLogy();
  hGroupX->Draw();
  cGroup->cd(2);
  gPad->SetLogy();
  hGroupZ->Draw();
  cGroup->Print("cGroup.pdf");
  cTotal->Write();
  cCommon->Write();
  cGroup->Write();
}
