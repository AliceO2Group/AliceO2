/// \file CheckDigits.C
/// \brief Simple macro to check ITSU clusters

#if !defined(__CLING__) || defined(__ROOTCLING__)
#include <TCanvas.h>
#include <TFile.h>
#include <TH2F.h>
#include <TNtuple.h>
#include <TString.h>
#include <TTree.h>

#include "ITSBase/GeometryTGeo.h"
#include "DataFormatsITSMFT/Cluster.h"
#include "ITSMFTSimulation/Hit.h"
#include "MathUtils/Cartesian3D.h"
#include "MathUtils/Utils.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#endif

void CheckClusters(Int_t nEvents = 10, TString mcEngine = "TGeant3")
{
  using namespace o2::Base;
  using namespace o2::ITS;

  using o2::ITSMFT::Cluster;
  using o2::ITSMFT::Hit;

  TFile* f = TFile::Open("CheckClusters.root", "recreate");
  TNtuple* nt = new TNtuple("ntc", "cluster ntuple", "x:y:z:dx:dz:lab:rof:ev:hlx:hlz:clx:clz");

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
  clusTree->SetBranchAddress("ITSCluster", &clusArr);
  // Cluster MC labels
  o2::dataformats::MCTruthContainer<o2::MCCompLabel>* clusLabArr = nullptr;
  clusTree->SetBranchAddress("ITSClusterMCTruth", &clusLabArr);

  Int_t nevCl = clusTree->GetEntries(); // clusters in cont. readout may be grouped as few events per entry
  Int_t nevH = hitTree->GetEntries();   // hits are stored as one event per entry
  int ievC = 0, ievH = 0;
  int lastReadHitEv = -1;
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
      }
      nt->Fill(gloC.X(), gloC.Y(), gloC.Z(), dx, dz, trID, c.getROFrame(), ievC, locH.X(), locH.Z(), locC.X(),
               locC.Z());
    }
  }
  new TCanvas;
  nt->Draw("y:x");
  new TCanvas;
  nt->Draw("dx:dz");
  f->cd();
  nt->Write();
  f->Close();
}
