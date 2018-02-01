/// \file CheckDigits.C
/// \brief Simple macro to check ITSU clusters

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

#include "DetectorsBase/Utils.h"
#include "ITSBase/GeometryTGeo.h"
#include "ITSMFTReconstruction/BuildTopologyDictionary.h"
#include "ITSMFTReconstruction/Cluster.h"
#include "ITSMFTReconstruction/ClusterTopology.h"
#include "ITSMFTSimulation/Hit.h"
#include "MathUtils/Cartesian3D.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/MCTruthContainer.h"

#endif

void CheckTopologies(Int_t nEvents = 10, TString mcEngine = "TGeant3")
{
  using namespace o2::Base;
  using namespace o2::ITS;

  using o2::ITSMFT::BuildTopologyDictionary;
  using o2::ITSMFT::Cluster;
  using o2::ITSMFT::ClusterTopology;
  using o2::ITSMFT::Hit;

  char filename[100];

  // Geometry
  sprintf(filename, "AliceO2_%s.params_%i.root", mcEngine.Data(), nEvents);
  TFile* file = TFile::Open(filename);
  gFile->Get("FairGeoParSet");

  auto gman = o2::ITS::GeometryTGeo::Instance();
  gman->fillMatrixCache(o2::utils::bit2Mask(o2::TransformType::T2L, o2::TransformType::T2GRot,
                                            o2::TransformType::L2G)); // request cached transforms

  // Hits
  sprintf(filename, "AliceO2_%s.mc_%i_event.root", mcEngine.Data(), nEvents);
  TFile* file0 = TFile::Open(filename);
  TTree* hitTree = (TTree*)gFile->Get("o2sim");
  std::vector<Hit>* hitArray = nullptr;
  hitTree->SetBranchAddress("ITSHit", &hitArray);

  // Clusters
  sprintf(filename, "AliceO2_%s.clus_%i_event.root", mcEngine.Data(), nEvents);
  TFile* file1 = TFile::Open(filename);
  TTree* clusTree = (TTree*)gFile->Get("o2sim");
  std::vector<Cluster>* clusArr = nullptr;
  // clusTree->SetBranchAddress("ITSCluster",&clusArr); // Why this does not work ???
  auto* branch = clusTree->GetBranch("ITSCluster");
  if (!branch) {
    std::cout << "No clusters !" << std::endl;
    return;
  }
  branch->SetAddress(&clusArr);
  // Cluster MC labels
  o2::dataformats::MCTruthContainer<o2::MCCompLabel>* clusLabArr = nullptr;
  clusTree->SetBranchAddress("ITSClusterMCTruth", &clusLabArr);

  Int_t nevCl = clusTree->GetEntries(); // clusters in cont. readout may be grouped as few events per entry
  Int_t nevH = hitTree->GetEntries();   // hits are stored as one event per entry
  int ievC = 0, ievH = 0;
  int lastReadHitEv = -1;

  // Topologies dictionaries: 1) all clusters 2) signal clusters only 3) noise clusters only
  BuildTopologyDictionary completeDictionary;
  BuildTopologyDictionary signalDictionary;
  BuildTopologyDictionary noiseDictionary;

  for (ievC = 0; ievC < nevCl; ievC++) {
    clusTree->GetEvent(ievC);
    Int_t nc = clusArr->size();
    printf("processing cluster event %d\n", ievC);

    while (nc--) {
      // cluster is in tracking coordinates always
      Cluster& c = (*clusArr)[nc];
      Int_t chipID = c.getSensorID();
      const auto locC = c.getXYZLoc(*gman);    // convert from tracking to local frame
      const auto gloC = c.getXYZGloRot(*gman); // convert from tracking to global frame
      auto lab = (clusLabArr->getLabels(nc))[0];

      int rowSpan = c.getPatternRowSpan();
      int columnSpan = c.getPatternColSpan();
      int nBytes = (rowSpan * columnSpan) >> 3;
      if (((rowSpan * columnSpan) % 8) != 0)
        nBytes++;
      unsigned char patt[Cluster::kMaxPatternBytes];
      c.getPattern(&patt[0], nBytes);
      ClusterTopology topology(rowSpan, columnSpan, patt);

      float dx = 0, dz = 0;
      int trID = lab.getTrackID();
      int ievH = lab.getEventID();
      Point3D<float> locH, locHsta;
      if (trID >= 0) { // is this cluster from hit or noise ?
        Hit* p = nullptr;
        if (lastReadHitEv != ievH) {
          hitTree->GetEvent(ievH);
          lastReadHitEv = ievH;
        }
        for (auto& ptmp : *hitArray) {
          if (ptmp.GetDetectorID() != chipID)
            continue;
          if (ptmp.GetTrackID() != trID)
            continue;
          p = &ptmp;
          break;
        }
        if (!p) {
          printf("did not find hit (scanned HitEvs %d %d) for cluster of tr%d on chip %d\n", ievH, nevH, trID, chipID);
          locH.SetXYZ(0.f, 0.f, 0.f);
        } else {
          // mean local position of the hit
          locH = gman->getMatrixL2G(chipID) ^ (p->GetPos()); // inverse conversion from global to local
          locHsta = gman->getMatrixL2G(chipID) ^ (p->GetPosStart());
          locH.SetXYZ(0.5 * (locH.X() + locHsta.X()), 0.5 * (locH.Y() + locHsta.Y()), 0.5 * (locH.Z() + locHsta.Z()));
          // std::cout << "chip "<< p->GetDetectorID() << "  PposGlo " << p->GetPos() << std::endl;
          // std::cout << "chip "<< c->getSensorID() << "  PposLoc " << locH << std::endl;
          dx = locH.X() - locC.X();
          dz = locH.Z() - locC.Z();
        }
        signalDictionary.accountTopology(topology, dx, dz);
      } else {
        noiseDictionary.accountTopology(topology, dx, dz);
      }
      completeDictionary.accountTopology(topology, dx, dz);
    }
  }
  completeDictionary.setThreshold(0.00001);
  completeDictionary.groupRareTopologies();
  completeDictionary.printDictionaryBinary("complete_dictionary.bin");
  ofstream completeDictionaryOutput("complete_dictionary.txt");
  completeDictionaryOutput << completeDictionary;
  noiseDictionary.setThreshold(0.00001);
  noiseDictionary.groupRareTopologies();
  noiseDictionary.printDictionaryBinary("noise_dictionary.bin");
  ofstream noiseDictionaryOutput("noise_dictionary.txt");
  noiseDictionaryOutput << noiseDictionary;
  signalDictionary.setThreshold(0.00001);
  signalDictionary.groupRareTopologies();
  signalDictionary.printDictionaryBinary("signal_dictionary.bin");
  ofstream signalDictionaryOutput("signal_dictionary.txt");
  signalDictionaryOutput << signalDictionary;

  TFile histogramOutput("histograms.root", "recreate");
  TCanvas* cComplete = new TCanvas("cComplete", "Distribution of all the topologies");
  cComplete->cd();
  cComplete->SetLogy();
  TH1F* hComplete = (TH1F*)completeDictionary.mHdist.Clone("hComplete");
  hComplete->SetDirectory(0);
  hComplete->SetTitle("Topology distribution");
  hComplete->GetXaxis()->SetTitle("Topology ID");
  hComplete->SetFillColor(kRed);
  hComplete->SetFillStyle(3005);
  hComplete->Draw("hist");
  cComplete->Write();
  TCanvas* cNoise = new TCanvas("cNoise", "Distribution of noise topologies");
  cNoise->cd();
  cNoise->SetLogy();
  TH1F* hNoise = (TH1F*)noiseDictionary.mHdist.Clone("hNoise");
  hNoise->SetDirectory(0);
  hNoise->SetTitle("Topology distribution");
  hNoise->GetXaxis()->SetTitle("Topology ID");
  hNoise->SetFillColor(kRed);
  hNoise->SetFillStyle(3005);
  hNoise->Draw("hist");
  cNoise->Write();
  TCanvas* cProper = new TCanvas("cProper", "cProper");
  cProper->cd();
  cProper->SetLogy();
  TH1F* hProper = (TH1F*)signalDictionary.mHdist.Clone("hProper");
  hProper->SetDirectory(0);
  hProper->SetTitle("Topology distribution");
  hProper->GetXaxis()->SetTitle("Topology ID");
  hProper->SetFillColor(kRed);
  hProper->SetFillStyle(3005);
  hProper->Draw("hist");
  cProper->Write();
}
