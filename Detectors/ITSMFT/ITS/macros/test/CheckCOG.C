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
#include "DetectorsCommonDataFormats/NameConf.h"

#endif

void CheckCOG(std::string clusfile = "o2clus_its.root", std::string inputGeom = "", std::string dictionary_file = "")
{
  gStyle->SetOptStat(0);
  using namespace o2::base;
  using namespace o2::its;

  using o2::itsmft::Cluster;
  using o2::itsmft::CompCluster;
  using o2::itsmft::CompClusterExt;
  using o2::itsmft::TopologyDictionary;

  if (dictionary_file.empty()) {
    dictionary_file = o2::base::NameConf::getDictionaryFileName(o2::detectors::DetID::ITS, "", ".bin");
  }

  // Geometry
  o2::base::GeometryManager::loadGeometry(inputGeom);
  auto gman = o2::its::GeometryTGeo::Instance();
  gman->fillMatrixCache(o2::utils::bit2Mask(o2::TransformType::T2L, o2::TransformType::T2GRot,
                                            o2::TransformType::L2G)); // request cached transforms

  // Clusters
  TFile* file1 = TFile::Open(clusfile.data());
  TTree* clusTree = (TTree*)gFile->Get("o2sim");
  std::vector<CompClusterExt>* compclusArr = nullptr;
  clusTree->SetBranchAddress("ITSClusterComp", &compclusArr);
  std::vector<unsigned char>* patternsPtr = nullptr;
  auto pattBranch = clusTree->GetBranch("ITSClusterPatt");
  if (pattBranch) {
    pattBranch->SetAddress(&patternsPtr);
  }

  int nevCl = clusTree->GetEntries(); // clusters in cont. readout may be grouped as few events per entry
  int ievC = 0;
  int lastReadHitEv = -1;

  TopologyDictionary dict;
  std::ifstream file(dictionary_file.c_str());
  if (file.good()) {
    LOG(INFO) << "Running with dictionary: " << dictionary_file.c_str();
    dict.readBinaryFile(dictionary_file);
  } else {
    LOG(INFO) << "Running without dictionary !";
  }

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
    int nc = compclusArr->size();
    printf("processing cluster event %d\n", ievC);

    std::vector<unsigned char>::const_iterator pattIdx;
    if (patternsPtr)
      pattIdx = patternsPtr->begin();
    for (int i = 0; i < nc; i++) {
      // cluster is in tracking coordinates always
      Cluster& c = (*clusArr)[i];
      const auto locC = c.getXYZLoc(*gman); // convert from tracking to local frame

      float dx, dz;
      Point3D<float> locComp;
      CompClusterExt& cComp = (*compclusArr)[i];
      auto pattID = cComp.getPatternID();
      fOut << Form("groupID: %d\n", pattID);
      if (pattID != CompCluster::InvalidPatternID) {
        Point3D<float> locComp = dict.getClusterCoordinates(cComp);

        float xComp = locComp.X();
        float zComp = locComp.Z();

        dx = (locC.X() - xComp) * 10000;
        dz = (locC.Z() - zComp) * 10000;

        hTotalX->Fill(dx);
        hTotalZ->Fill(dz);

        fOut << Form("groupID: %d\n", cComp.getPatternID());
        fOut << Form("is group: %o\n", dict.isGroup(cComp.getPatternID()));
        fOut << Form("x_full: %.4f x_comp: %.4f dx: %.4f x_shift: %.4f\n", locC.X(), xComp, dx / 10000, dict.getXCOG(cComp.getPatternID()));
        fOut << Form("z_full: %.4f z_comp: %.4f dZ: %.4f z_shift: %.4f\n", locC.Z(), zComp, dz / 10000, dict.getZCOG(cComp.getPatternID()));
        fOut << dict.getPattern(cComp.getPatternID());
        fOut << Form("***************************************\n");

        if (dict.isGroup(cComp.getPatternID())) {
          if (patternsPtr) { // Restore the full pixel pattern information from the auxiliary branch
            o2::itsmft::ClusterPattern patt(pattIdx);
            auto locCl = dict.getClusterCoordinates(cComp, patt);
            dx = (locC.X() - locCl.X()) * 10000;
            dz = (locC.Z() - locCl.Z()) * 10000;
          }
          hGroupX->Fill(dx);
          hGroupZ->Fill(dz);
        } else {
          hCommonX->Fill(dx);
          hCommonZ->Fill(dz);
        }
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
