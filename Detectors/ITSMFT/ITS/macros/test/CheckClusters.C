/// \file CheckClusters.C
/// \brief Simple macro to check ITSU clusters

#if !defined(__CLING__) || defined(__ROOTCLING__)
#include <TCanvas.h>
#include <TFile.h>
#include <TH2F.h>
#include <TNtuple.h>
#include <TString.h>
#include <TTree.h>

#include "ITSMFTBase/SegmentationAlpide.h"
#include "ITSBase/GeometryTGeo.h"
#include "DataFormatsITSMFT/CompCluster.h"
#include "DataFormatsITSMFT/TopologyDictionary.h"
#include "ITSMFTSimulation/Hit.h"
#include "DataFormatsITSMFT/ROFRecord.h"
#include "MathUtils/Cartesian.h"
#include "MathUtils/Utils.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "DetectorsCommonDataFormats/NameConf.h"
#endif

void CheckClusters(std::string clusfile = "o2clus_its.root", std::string hitfile = "o2sim_HitsITS.root",
                   std::string inputGeom = "", std::string paramfile = "o2sim_par.root",
                   std::string dictfile = "")
{
  const int QEDSourceID = 99; // Clusters from this MC source correspond to QED electrons

  using namespace o2::base;
  using namespace o2::its;

  using Segmentation = o2::itsmft::SegmentationAlpide;
  using o2::itsmft::CompClusterExt;
  using o2::itsmft::Hit;
  using ROFRec = o2::itsmft::ROFRecord;
  using MC2ROF = o2::itsmft::MC2ROFRecord;
  using HitVec = std::vector<Hit>;
  using MC2HITS_map = std::unordered_map<uint64_t, int>; // maps (track_ID<<16 + chip_ID) to entry in the hit vector

  std::vector<HitVec*> hitVecPool;
  std::vector<MC2HITS_map> mc2hitVec;

  TFile fout("CheckClusters.root", "recreate");
  TNtuple nt("ntc", "cluster ntuple", "ev:lab:hlx:hlz:tx:tz:cgx:cgy:cgz:dx:dy:dz:ex:ez:patid:rof:npx:id");

  // Geometry
  o2::base::GeometryManager::loadGeometry(inputGeom);
  auto gman = o2::its::GeometryTGeo::Instance();
  gman->fillMatrixCache(o2::math_utils::bit2Mask(o2::math_utils::TransformType::T2L, o2::math_utils::TransformType::T2GRot,
                                                 o2::math_utils::TransformType::L2G)); // request cached transforms

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
  std::vector<CompClusterExt>* clusArr = nullptr;
  clusTree->SetBranchAddress("ITSClusterComp", &clusArr);
  std::vector<unsigned char>* patternsPtr = nullptr;
  auto pattBranch = clusTree->GetBranch("ITSClusterPatt");
  if (pattBranch) {
    pattBranch->SetAddress(&patternsPtr);
  }
  if (dictfile.empty()) {
    dictfile = o2::base::NameConf::getAlpideClusterDictionaryFileName(o2::detectors::DetID::ITS, "", "bin");
  }
  o2::itsmft::TopologyDictionary dict;
  std::ifstream file(dictfile.c_str());
  if (file.good()) {
    LOG(INFO) << "Running with dictionary: " << dictfile.c_str();
    dict.readBinaryFile(dictfile);
  } else {
    LOG(INFO) << "Running without dictionary !";
  }

  // ROFrecords
  std::vector<ROFRec> rofRecVec, *rofRecVecP = &rofRecVec;
  clusTree->SetBranchAddress("ITSClustersROF", &rofRecVecP);

  // Cluster MC labels
  o2::dataformats::MCTruthContainer<o2::MCCompLabel>* clusLabArr = nullptr;
  std::vector<MC2ROF> mc2rofVec, *mc2rofVecP = &mc2rofVec;
  if (hitTree && clusTree->GetBranch("ITSClusterMCTruth")) {
    clusTree->SetBranchAddress("ITSClusterMCTruth", &clusLabArr);
    clusTree->SetBranchAddress("ITSClustersMC2ROF", &mc2rofVecP);
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
  auto pattIt = patternsPtr->cbegin();
  for (int irof = 0; irof < nROFRec; irof++) {
    const auto& rofRec = rofRecVec[irof];

    rofRec.print();

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
    for (int icl = 0; icl < rofRec.getNEntries(); icl++) {
      int clEntry = rofRec.getFirstEntry() + icl; // entry of icl-th cluster of this ROF in the vector of clusters

      const auto& cluster = (*clusArr)[clEntry];

      float errX{0.f};
      float errZ{0.f};
      int npix = 0;
      auto pattID = cluster.getPatternID();
      o2::math_utils::Point3D<float> locC;
      if (pattID == o2::itsmft::CompCluster::InvalidPatternID || dict.isGroup(pattID)) {
        o2::itsmft::ClusterPattern patt(pattIt);
        locC = dict.getClusterCoordinates(cluster, patt, false);
      } else {
        locC = dict.getClusterCoordinates(cluster);
        errX = dict.getErrX(pattID);
        errZ = dict.getErrZ(pattID);
        npix = dict.getNpixels(pattID);
      }
      auto chipID = cluster.getSensorID();
      // Transformation to the local --> global
      auto gloC = gman->getMatrixL2G(chipID) * locC;

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
      float dx = 0, dz = 0;
      int ievH = lab.getEventID();
      o2::math_utils::Point3D<float> locH, locHsta;

      // mean local position of the hit
      locH = gman->getMatrixL2G(chipID) ^ (hit.GetPos()); // inverse conversion from global to local
      locHsta = gman->getMatrixL2G(chipID) ^ (hit.GetPosStart());
      auto x0 = locHsta.X(), dltx = locH.X() - x0;
      auto y0 = locHsta.Y(), dlty = locH.Y() - y0;
      auto z0 = locHsta.Z(), dltz = locH.Z() - z0;
      auto r = (0.5 * (Segmentation::SensorLayerThickness - Segmentation::SensorLayerThicknessEff) - y0) / dlty;
      locH.SetXYZ(x0 + r * dltx, y0 + r * dlty, z0 + r * dltz);
      //locH.SetXYZ(0.5 * (locH.X() + locHsta.X()), 0.5 * (locH.Y() + locHsta.Y()), 0.5 * (locH.Z() + locHsta.Z()));
      std::array<float, 18> data = {(float)lab.getEventID(), (float)trID,
                                    locH.X(), locH.Z(), dltx / dlty, dltz / dlty,
                                    gloC.X(), gloC.Y(), gloC.Z(),
                                    locC.X() - locH.X(), locC.Y() - locH.Y(), locC.Z() - locH.Z(),
                                    errX, errZ, (float)pattID,
                                    (float)rofRec.getROFrame(), (float)npix, (float)chipID};
      nt.Fill(data.data());
    }
  }

  new TCanvas;
  nt.Draw("cgy:cgx");
  new TCanvas;
  nt.Draw("dz:dx", "abs(dz)<0.01 && abs(dx)<0.01");
  new TCanvas;
  nt.Draw("dz:tz", "abs(dz)<0.005 && abs(tz)<2");

  auto c1 = new TCanvas("p1", "pullX");
  c1->cd();
  c1->SetLogy();
  nt.Draw("dx/ex", "abs(dx/ex)<10&&patid<10");
  auto c2 = new TCanvas("p2", "pullZ");
  c2->cd();
  c2->SetLogy();
  nt.Draw("dz/ez", "abs(dz/ez)<10&&patid<10");

  auto d1 = new TCanvas("d1", "deltaX");
  d1->cd();
  d1->SetLogy();
  nt.Draw("dx", "abs(dx)<5");
  auto d2 = new TCanvas("d2", "deltaZ");
  d2->cd();
  d2->SetLogy();
  nt.Draw("dz", "abs(dz)<5");

  fout.cd();
  nt.Write();
}
