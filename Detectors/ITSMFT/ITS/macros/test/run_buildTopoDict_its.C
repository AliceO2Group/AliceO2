/// \file run_buildTopoDict_its
/// Macros to generate dictionary of topologies.

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
#include <TStopwatch.h>
#include <TSystem.h>
#include <fstream>
#include <string>

#include "MathUtils/Utils.h"
#include "ITSBase/GeometryTGeo.h"
#include "ITSMFTReconstruction/BuildTopologyDictionary.h"
#include "DataFormatsITSMFT/Cluster.h"
#include "DataFormatsITSMFT/ClusterTopology.h"
#include "DataFormatsITSMFT/ROFRecord.h"
#include "ITSMFTSimulation/Hit.h"
#include "MathUtils/Cartesian3D.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include <unordered_map>
#endif

/// Build dictionary of topologies from the root file with full clusters
/// If the hitfile is non-empty, the mean bias between the cluster COG
/// and mean MC hit position is calculated

void run_buildTopoDict_its(std::string clusfile = "o2clus_its.root",
                           std::string hitfile = "o2sim.root",
                           std::string inputGeom = "O2geometry.root")
{
  const int QEDSourceID = 99; // Clusters from this MC source correspond to QED electrons

  using namespace o2::base;
  using namespace o2::its;

  using o2::itsmft::BuildTopologyDictionary;
  using o2::itsmft::Cluster;
  using o2::itsmft::ClusterTopology;
  using o2::itsmft::Hit;
  using ROFRec = o2::itsmft::ROFRecord;
  using MC2ROF = o2::itsmft::MC2ROFRecord;
  using HitVec = std::vector<Hit>;
  using MC2HITS_map = std::unordered_map<uint64_t, int>; // maps (track_ID<<16 + chip_ID) to entry in the hit vector

  std::vector<HitVec*> hitVecPool;
  std::vector<MC2HITS_map> mc2hitVec;

  TStopwatch sw;
  sw.Start();

  // Geometry
  o2::base::GeometryManager::loadGeometry(inputGeom);
  auto gman = o2::its::GeometryTGeo::Instance();
  gman->fillMatrixCache(o2::utils::bit2Mask(o2::TransformType::T2L, o2::TransformType::T2GRot,
                                            o2::TransformType::L2G)); // request cached transforms

  // Hits if requested
  TFile* fileH = nullptr;
  TTree* hitTree = nullptr;

  if (!hitfile.empty() && !gSystem->AccessPathName(hitfile.c_str())) {
    fileH = TFile::Open(hitfile.data());
    hitTree = (TTree*)fileH->Get("o2sim");
    mc2hitVec.resize(hitTree->GetEntries());
    hitVecPool.resize(hitTree->GetEntries(), nullptr);
  }

  // Clusters
  TFile* fileCl = TFile::Open(clusfile.data());
  TTree* clusTree = (TTree*)fileCl->Get("o2sim");
  std::vector<Cluster>* clusArr = nullptr;
  clusTree->SetBranchAddress("ITSCluster", &clusArr);

  // ROFrecords
  std::vector<ROFRec> rofRecVec, *rofRecVecP = &rofRecVec;
  TTree* ROFRecTree = (TTree*)fileCl->Get("ITSClustersROF");
  ROFRecTree->SetBranchAddress("ITSClustersROF", &rofRecVecP);

  // Cluster MC labels
  o2::dataformats::MCTruthContainer<o2::MCCompLabel>* clusLabArr = nullptr;
  std::vector<MC2ROF> mc2rofVec, *mc2rofVecP = &mc2rofVec;
  TTree* MC2ROFRecTree = nullptr;
  if (hitTree && clusTree->GetBranch("ITSClusterMCTruth")) {
    clusTree->SetBranchAddress("ITSClusterMCTruth", &clusLabArr);
    MC2ROFRecTree = (TTree*)fileCl->Get("ITSClustersMC2ROF");
    MC2ROFRecTree->SetBranchAddress("ITSClustersMC2ROF", &mc2rofVecP);
  }

  BuildTopologyDictionary dict;

  for (int rofEnt = 0; rofEnt < ROFRecTree->GetEntries(); rofEnt++) {
    ROFRecTree->GetEntry(rofEnt);
    int nROFRec = (int)rofRecVec.size();
    std::vector<int> mcEvMin(nROFRec, hitTree->GetEntries()), mcEvMax(nROFRec, -1);

    if (clusLabArr) { // >> build min and max MC events used by each ROF
      for (int ent = 0; ent < MC2ROFRecTree->GetEntries(); ent++) {
        MC2ROFRecTree->GetEntry(ent);
        for (int imc = mc2rofVec.size(); imc--;) {
          const auto& mc2rof = mc2rofVec[imc];
          if (mc2rof.rofRecordID < 0) {
            continue; // this MC event did not contribute to any ROF
          }
          for (int irfd = mc2rof.maxROF - mc2rof.minROF + 1; irfd--;) {
            int irof = mc2rof.rofRecordID + irfd;
            if (mcEvMin[irof] > imc) {
              mcEvMin[irof] = imc;
            }
            if (mcEvMax[irof] < imc) {
              mcEvMax[irof] = imc;
            }
          }
        }
      }
    } // << build min and max MC events used by each ROF

    for (int irof = 0; irof < nROFRec; irof++) {
      const auto& rofRec = rofRecVec[irof];

      rofRec.print();
      if (clusTree->GetReadEntry() != rofRec.getROFEntry().getEvent()) { // read the entry containing clusters of given ROF
        clusTree->GetEntry(rofRec.getROFEntry().getEvent());             // all clusters of the same ROF are in a single entry
      }

      if (clusLabArr) { // >> read and map MC events contributing to this ROF
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
      } // << cache MC events contributing to this ROF

      for (int icl = 0; icl < rofRec.getNROFEntries(); icl++) {
        int clEntry = rofRec.getROFEntry().getIndex() + icl; // entry of icl-th cluster of this ROF in the vector of clusters
        // do we read MC data?

        const auto& cluster = (*clusArr)[clEntry];

        int rowSpan = cluster.getPatternRowSpan();
        int columnSpan = cluster.getPatternColSpan();
        int nBytes = (rowSpan * columnSpan) >> 3;
        if (((rowSpan * columnSpan) % 8) != 0) {
          nBytes++;
        }
        unsigned char patt[Cluster::kMaxPatternBytes];
        cluster.getPattern(&patt[0], nBytes);
        ClusterTopology topology(rowSpan, columnSpan, patt);
        //
        // do we need to account for the bias of cluster COG wrt MC hit center?
        float dX = BuildTopologyDictionary::IgnoreVal, dZ = BuildTopologyDictionary::IgnoreVal;
        if (clusLabArr) {
          const auto& lab = (clusLabArr->getLabels(clEntry))[0]; // we neglect effect of cluster contributed by multiple hits
          auto srcID = lab.getSourceID();
          if (!lab.isNoise() && srcID != QEDSourceID) { // use MC truth info only for non-QED and non-noise clusters
            auto trID = lab.getTrackID();
            const auto& mc2hit = mc2hitVec[lab.getEventID()];
            const auto* hitArray = hitVecPool[lab.getEventID()];
            int chipID = cluster.getSensorID();
            uint64_t key = (uint64_t(trID) << 32) + chipID;
            auto hitEntry = mc2hit.find(key);
            if (hitEntry != mc2hit.end()) {
              const auto& hit = (*hitArray)[hitEntry->second];
              auto locH = gman->getMatrixL2G(chipID) ^ (hit.GetPos()); // inverse conversion from global to local
              auto locHsta = gman->getMatrixL2G(chipID) ^ (hit.GetPosStart());
              locH.SetXYZ(0.5 * (locH.X() + locHsta.X()), 0.5 * (locH.Y() + locHsta.Y()), 0.5 * (locH.Z() + locHsta.Z()));
              const auto locC = cluster.getXYZLoc(*gman); // convert from tracking to local frame
              dX = locH.X() - locC.X();
              dZ = locH.Z() - locC.Z();
            } else {
              printf("Failed to find MC hit entry for Tr:%d chipID:%d\n", trID, chipID);
            }
          }
        }
        dict.accountTopology(topology, dX, dZ);
      }
      // clean MC cache for events which are not needed anymore
      int irfNext = irof;
      while ((++irfNext < nROFRec) && mcEvMax[irfNext] < 0) {
      }                                                                      // find next ROF having MC contribution
      int limMC = irfNext == nROFRec ? hitVecPool.size() : mcEvMin[irfNext]; // can clean events up to this
      for (int imc = mcEvMin[irof]; imc < limMC; imc++) {
        delete hitVecPool[imc];
        hitVecPool[imc] = nullptr;
        mc2hitVec[imc].clear();
      }
    }
  }

  dict.setThreshold(0.0001);
  dict.groupRareTopologies();
  dict.printDictionaryBinary("dictionary.bin");
  dict.printDictionary("dictionary.txt");
  dict.saveDictionaryRoot("dictionary.root");

  TFile histogramOutput("dict_histograms.root", "recreate");
  TCanvas* cComplete = new TCanvas("cComplete", "Distribution of all the topologies");
  cComplete->cd();
  cComplete->SetLogy();
  TH1F* hComplete = (TH1F*)dict.mHdist.Clone("hDict");
  hComplete->SetDirectory(0);
  hComplete->SetTitle("Topology distribution");
  hComplete->GetXaxis()->SetTitle("Topology ID");
  hComplete->SetFillColor(kRed);
  hComplete->SetFillStyle(3005);
  hComplete->Draw("hist");
  cComplete->Print("dictHisto.pdf");
  cComplete->Write();
  sw.Stop();
  sw.Print();
}
