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
#include "DataFormatsITSMFT/ROFRecord.h"
#include "MathUtils/Cartesian3D.h"
#include "MathUtils/Utils.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#endif

void CheckClusters(std::string clusfile = "o2clus_its.root", std::string hitfile = "o2sim.root", std::string inputGeom = "O2geometry.root", std::string paramfile = "o2sim_par.root")
{
  const int QEDSourceID = 99; // Clusters from this MC source correspond to QED electrons

  using namespace o2::base;
  using namespace o2::its;

  using o2::itsmft::Cluster;
  using o2::itsmft::Hit;
  using ROFRec = o2::itsmft::ROFRecord;
  using MC2ROF = o2::itsmft::MC2ROFRecord;
  using HitVec = std::vector<Hit>;
  using MC2HITS_map = std::unordered_map<uint64_t, int>; // maps (track_ID<<16 + chip_ID) to entry in the hit vector

  std::vector<HitVec*> hitVecPool;
  std::vector<MC2HITS_map> mc2hitVec;

  TFile fout("CheckClusters.root", "recreate");
  TNtuple nt("ntc", "cluster ntuple", "x:y:z:dx:dz:lab:ev:rof:hlx:hlz:clx:clz:n:id");

  // Geometry
  o2::base::GeometryManager::loadGeometry(inputGeom, "FAIRGeom");
  auto gman = o2::its::GeometryTGeo::Instance();
  gman->fillMatrixCache(o2::utils::bit2Mask(o2::TransformType::T2L, o2::TransformType::T2GRot,
                                            o2::TransformType::L2G)); // request cached transforms

  // Hits
  TFile fileH(hitfile.data());
  TTree* hitTree = (TTree*)fileH.Get("o2sim");
  std::vector<o2::itsmft::Hit>* hitArray = nullptr;
  hitTree->SetBranchAddress("ITSHit", &hitArray);
  mc2hitVec.resize(hitTree->GetEntries());
  hitVecPool.resize(hitTree->GetEntries(), nullptr);

  // Clusters
  TFile fileC(clusfile.data());
  TTree* clusTree = (TTree*)fileC.Get("o2sim");
  std::vector<Cluster>* clusArr = nullptr;
  clusTree->SetBranchAddress("ITSCluster", &clusArr);

  // ROFrecords
  std::vector<ROFRec> rofRecVec, *rofRecVecP = &rofRecVec;
  TTree* ROFRecTree = (TTree*)fileC.Get("ITSClustersROF");
  ROFRecTree->SetBranchAddress("ITSClustersROF", &rofRecVecP);

  // Cluster MC labels
  o2::dataformats::MCTruthContainer<o2::MCCompLabel>* clusLabArr = nullptr;
  std::vector<MC2ROF> mc2rofVec, *mc2rofVecP = &mc2rofVec;
  TTree* MC2ROFRecTree = nullptr;
  if (hitTree && clusTree->GetBranch("ITSClusterMCTruth")) {
    clusTree->SetBranchAddress("ITSClusterMCTruth", &clusLabArr);
    MC2ROFRecTree = (TTree*)fileC.Get("ITSClustersMC2ROF");
    MC2ROFRecTree->SetBranchAddress("ITSClustersMC2ROF", &mc2rofVecP);
  }

  ROFRecTree->GetEntry(0);
  int nROFRec = (int)rofRecVec.size();
  std::vector<int> mcEvMin(nROFRec, hitTree->GetEntries());
  std::vector<int> mcEvMax(nROFRec, -1);

  // >> build min and max MC events used by each ROF
  for (int ent = 0; ent < MC2ROFRecTree->GetEntries(); ent++) {
    MC2ROFRecTree->GetEntry(ent);
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
  }
  // << build min and max MC events used by each ROF

  for (int irof = 0; irof < nROFRec; irof++) {
    const auto& rofRec = rofRecVec[irof];

    rofRec.print();
    if (clusTree->GetReadEntry() != rofRec.getROFEntry().getEvent()) { // read the entry containing clusters of given ROF
      clusTree->GetEntry(rofRec.getROFEntry().getEvent());             // all clusters of the same ROF are in a single entry
    }
    // >> read and map MC events contributing to this ROF
    for (int im = mcEvMin[irof]; im <= mcEvMax[irof]; im++) {
      if (!hitVecPool[im]) {
        hitTree->SetBranchAddress("ITSHit", &hitVecPool[im]);
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
    for (int icl = 0; icl < rofRec.getNROFEntries(); icl++) {
      int clEntry = rofRec.getROFEntry().getIndex() + icl; // entry of icl-th cluster of this ROF in the vector of clusters

      const auto& cluster = (*clusArr)[clEntry];

      int chipID = cluster.getSensorID();
      const auto locC = cluster.getXYZLoc(*gman);    // convert from tracking to local frame
      const auto gloC = cluster.getXYZGloRot(*gman); // convert from tracking to global frame
      const auto& lab = (clusLabArr->getLabels(clEntry))[0];

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
      nt.Fill(gloC.X(), gloC.Y(), gloC.Z(), dx, dz, trID, lab.getEventID(),
              rofRec.getROFrame(), locH.X(), locH.Z(), locC.X(), locC.Z(), npix, chipID);
    }
  }

  new TCanvas;
  nt.Draw("y:x");
  new TCanvas;
  nt.Draw("dx:dz");
  fout.cd();
  nt.Write();
}
