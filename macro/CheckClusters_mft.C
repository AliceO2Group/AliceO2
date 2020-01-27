/// \file CheckClusters_mft.C
/// \brief Simple macro to check MFT clusters

#if !defined(__CLING__) || defined(__ROOTCLING__)

#include <TCanvas.h>
#include <TFile.h>
#include <TH2F.h>
#include <TNtuple.h>
#include <TString.h>
#include <TTree.h>

#include "DataFormatsITSMFT/Cluster.h"
#include "ITSMFTSimulation/Hit.h"
#include "DataFormatsITSMFT/ROFRecord.h"
#include "MFTBase/GeometryTGeo.h"
#include "MathUtils/Cartesian3D.h"
#include "MathUtils/Utils.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/MCTruthContainer.h"

#endif

using namespace o2::base;
using o2::itsmft::Cluster;

void CheckClusters_mft(Int_t nEvents = 10, Int_t nMuons = 200)
{
  const int QEDSourceID = 99; // Clusters from this MC source correspond to QED electrons

  using namespace o2::base;
  using namespace o2::mft;

  using o2::itsmft::Hit;
  using ROFRec = o2::itsmft::ROFRecord;
  using MC2ROF = o2::itsmft::MC2ROFRecord;
  using HitVec = std::vector<Hit>;
  using MC2HITS_map = std::unordered_map<uint64_t, int>; // maps (track_ID<<16 + chip_ID) to entry in the hit vector

  std::vector<HitVec*> hitVecPool;
  std::vector<MC2HITS_map> mc2hitVec;

  TH1F* hTrackID = new TH1F("hTrackID", "hTrackID", 1.1 * nMuons + 1, -0.5, (nMuons + 0.1 * nMuons) + 0.5);
  TH2F* hDifLocXrZc = new TH2F("hDifLocXrZc", "hDifLocXrZc", 100, -50., +50., 100, -50., +50.);

  TFile fout("CheckClusters.root", "recreate");
  TNtuple nt("ntc", "cluster ntuple", "x:y:z:dx:dz:lab:ev:rof:hlx:hlz:clx:clz:n:id");

  Int_t nNoise = 0;
  Char_t filename[100];

  // Geometry
  o2::base::GeometryManager::loadGeometry("O2geometry.root", "FAIRGeom");
  auto gman = o2::mft::GeometryTGeo::Instance();
  gman->fillMatrixCache(o2::utils::bit2Mask(o2::TransformType::T2L, o2::TransformType::T2G,
                                            o2::TransformType::L2G)); // request cached transforms

  // Hits
  sprintf(filename, "o2sim.root");
  TFile* fileH = TFile::Open(filename);
  TTree* hitTree = (TTree*)gFile->Get("o2sim");
  std::vector<Hit>* hitArray = nullptr;
  hitTree->SetBranchAddress("MFTHit", &hitArray);
  mc2hitVec.resize(hitTree->GetEntries());
  hitVecPool.resize(hitTree->GetEntries(), nullptr);

  // Clusters
  sprintf(filename, "mftclusters.root");
  TFile* fileC = TFile::Open(filename);
  TTree* clusTree = (TTree*)gFile->Get("o2sim");
  std::vector<Cluster>* clusArray = nullptr;
  clusTree->SetBranchAddress("MFTCluster", &clusArray);

  // ROFrecords
  std::vector<ROFRec> rofRecVec, *rofRecVecP = &rofRecVec;
  clusTree->SetBranchAddress("MFTClustersROF", &rofRecVecP);

  // Cluster MC labels
  o2::dataformats::MCTruthContainer<o2::MCCompLabel>* clusLabArray = nullptr;
  std::vector<MC2ROF> mc2rofVec, *mc2rofVecP = &mc2rofVec;
  if (hitTree && clusTree->GetBranch("MFTClusterMCTruth")) {
    clusTree->SetBranchAddress("MFTClusterMCTruth", &clusLabArray);
    clusTree->SetBranchAddress("MFTClustersMC2ROF", &mc2rofVecP);
  }

  clusTree->GetEntry(0);
  int nROFRec = (int)rofRecVec.size();
  std::vector<int> mcEvMin(nROFRec, hitTree->GetEntries());
  std::vector<int> mcEvMax(nROFRec, -1);

  // >> build min and max MC events used by each ROF
  for (int imc = mc2rofVec.size(); imc--;) {
    const auto& mc2rof = mc2rofVec[imc];
    printf("MCRecord: ");
    mc2rof.print();
    if (mc2rof.rofRecordID < 0) {
      continue; // this MC event did not contribute to any ROF
    }
    for (int irfd = mc2rof.maxROF - mc2rof.minROF + 1; irfd--;) {
      int irof = mc2rof.rofRecordID + irfd;
      if (irof >= nROFRec) {
        LOG(ERROR) << "ROF=" << irof << " from MC2ROF record is >= N ROFs=" << nROFRec;
      }
      if (mcEvMin[irof] > imc) {
        mcEvMin[irof] = imc;
      }
      if (mcEvMax[irof] < imc) {
        mcEvMax[irof] = imc;
      }
    }
  }
  // << build min and max MC events used by each ROF

  for (int irof = 0; irof < nROFRec; irof++) {
    const auto& rofRec = rofRecVec[irof];

    rofRec.print();
    // >> read and map MC events contributing to this ROF
    for (int im = mcEvMin[irof]; im <= mcEvMax[irof]; im++) {
      if (!hitVecPool[im]) {
        hitTree->SetBranchAddress("MFTHit", &hitVecPool[im]);
        hitTree->GetEntry(im);
        auto& mc2hit = mc2hitVec[im];
        const auto* hitArray = hitVecPool[im];
        for (int ih = hitArray->size(); ih--;) {
          const auto& hit = (*hitArray)[ih];
          uint64_t key = (uint64_t(hit.GetTrackID()) << 32) + hit.GetDetectorID();
          mc2hit.emplace(key, ih);
        }
      }
    }
    // << cache MC events contributing to this ROF
    for (int icl = 0; icl < rofRec.getNEntries(); icl++) {
      int clEntry = rofRec.getFirstEntry() + icl; // entry of icl-th cluster of this ROF in the vector of clusters

      const auto& cluster = (*clusArray)[clEntry];

      int chipID = cluster.getSensorID();
      const auto locC = cluster.getXYZLoc(*gman); // convert from tracking to local frame
      const auto gloC = cluster.getXYZGlo(*gman); // convert from tracking to global frame
      const auto& lab = (clusLabArray->getLabels(clEntry))[0];

      if (!lab.isValid())
        nNoise++;

      if (!lab.isValid() || lab.getSourceID() == QEDSourceID)
        continue;

      // get MC info
      int trID = lab.getTrackID();
      const auto& mc2hit = mc2hitVec[lab.getEventID()];
      const auto* hitArray = hitVecPool[lab.getEventID()];
      uint64_t key = (uint64_t(trID) << 32) + chipID;
      auto hitEntry = mc2hit.find(key);
      if (hitEntry == mc2hit.end()) {
        LOG(ERROR) << "Failed to find MC hit entry for Tr" << trID << " chipID" << chipID;
        continue;
      }
      const auto& hit = (*hitArray)[hitEntry->second];
      hTrackID->Fill((Float_t)hit.GetTrackID());
      //
      int npix = cluster.getNPix();
      float dx = 0, dz = 0;
      int ievH = lab.getEventID();
      Point3D<float> locH, locHsta;

      // mean local position of the hit
      locH = gman->getMatrixL2G(chipID) ^ (hit.GetPos()); // inverse conversion from global to local
      locHsta = gman->getMatrixL2G(chipID) ^ (hit.GetPosStart());
      locH.SetXYZ(0.5 * (locH.X() + locHsta.X()), 0.5 * (locH.Y() + locHsta.Y()), 0.5 * (locH.Z() + locHsta.Z()));
      //std::cout << "chip "<< hit.GetDetectorID() << "  PposGlo " << hit.GetPos() << std::endl;
      //std::cout << "chip "<< chipID << "  PposLoc " << locH << std::endl;
      dx = locH.X() - locC.X();
      dz = locH.Z() - locC.Z();
      hDifLocXrZc->Fill(1.e4 * dx, 1.e4 * dz);
      nt.Fill(gloC.X(), gloC.Y(), gloC.Z(), dx, dz, trID, ievH,
              rofRec.getROFrame(), locH.X(), locH.Z(), locC.X(), locC.Z(), npix, chipID);
    }
  }

  printf("ntuple has %lld entries\n", nt.GetEntriesFast());

  TCanvas* c1 = new TCanvas("c1", "hTrackID", 50, 50, 600, 600);
  hTrackID->Scale(1. / (Float_t)nEvents);
  hTrackID->SetMinimum(0.);
  hTrackID->DrawCopy();

  TCanvas* c2 = new TCanvas("c2", "hDifLocXrZc", 50, 50, 600, 600);
  hDifLocXrZc->DrawCopy("COL2");

  new TCanvas;
  nt.Draw("y:x");
  new TCanvas;
  nt.Draw("dx:dz");
  fout.cd();
  nt.Write();
  hTrackID->Write();
  fout.Close();

  printf("noise clusters %d \n", nNoise);

  return;
}
