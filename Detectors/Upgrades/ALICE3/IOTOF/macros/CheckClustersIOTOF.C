// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file CheckClusters.C
/// \brief Simple macro to check TF3 clusters

#include <TCanvas.h>
#include <TFile.h>
#include <TH2F.h>
#include <TNtuple.h>
#include <TString.h>
#include <TTree.h>
#include <TLine.h>
#include <TStyle.h>

#include "IOTOFBase/Segmentation.h"
#include "IOTOFBase/GeometryTGeo.h"
#include "DataFormatsIOTOF/Cluster.h"
#include "IOTOFReconstruction/TopologyClassifier.h"
#include "ITSMFTSimulation/Hit.h"
#include "DetectorsBase/GeometryManager.h"
#if !defined(__CLING__) || defined(__ROOTCLING__)
#include <TCanvas.h>
#include <TFile.h>
#include <TH2F.h>
#include <TNtuple.h>
#include <TString.h>
#include <TTree.h>

#include "IOTOFBase/IOTOFBaseParam.h"
#include "IOTOFBase/GeometryTGeo.h"
#include "DataFormatsIOTOF/Cluster.h"
#include "IOTOFReconstruction/TopologyClassifier.h"
#include "ITSMFTSimulation/Hit.h"
#include "DataFormatsITSMFT/ROFRecord.h"
#include "MathUtils/Cartesian.h"
#include "MathUtils/Utils.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "DetectorsCommonDataFormats/DetectorNameConf.h"
#include "CCDB/BasicCCDBManager.h"
#endif

#define ENABLE_UPGRADES

void addTLines(float pitch)
{
  // Add grid lines at multiples of pitch on the current pad
  if (!gPad)
    return;

  gPad->Update();

  Double_t xmin = gPad->GetUxmin();
  Double_t xmax = gPad->GetUxmax();
  Double_t ymin = gPad->GetUymin();
  Double_t ymax = gPad->GetUymax();

  // Calculate the first vertical line position (multiple of pitch)
  int nLinesX = 0;
  for (float x = xmin; x <= xmax && nLinesX < 1000; x += pitch, nLinesX++) {
    TLine* line = new TLine(x, ymin, x, ymax);
    line->SetLineStyle(2);
    line->SetLineColor(kGray);
    line->Draw("same");
  }

  // Calculate the first horizontal line position (multiple of pitch)
  int nLinesY = 0;
  for (float y = ymin; y <= ymax && nLinesY < 1000; y += pitch, nLinesY++) {
    TLine* line = new TLine(xmin, y, xmax, y);
    line->SetLineStyle(2);
    line->SetLineColor(kGray);
    line->Draw("same");
  }

  gPad->Modified();
  gPad->Update();
}

void CheckClustersIOTOF(std::string clusfile = "tf3clusters.root",
                        std::string hitfile = "o2sim_HitsTF3.root",
                        std::string topodictfile = "TF3ClusterTopologies.root",
                        std::string inputGeom = "",
                        std::string cfgStr = "IOTOFBase.segmentedInnerTOF=true;IOTOFBase.segmentedOuterTOF=true;IOTOFBase.enableForwardTOF=false;IOTOFBase.enableBackwardTOF=false;")
{
  std::cout << "CheckClustersIOTOF: clusfile=" << clusfile << ", hitfile=" << hitfile << ", inputGeom=" << inputGeom << std::endl;
  const int QEDSourceID = 99; // Clusters from this MC source correspond to QED electrons

  using namespace o2::base;
  using namespace o2::iotof;

  using o2::iotof::Cluster;
  using o2::itsmft::Hit;

  o2::conf::ConfigurableParam::updateFromString(cfgStr);
  const auto& chipInfo = o2::iotof::ChipSpecificsParam::Instance();
  auto seg = o2::iotof::Segmentation::Instance();

  using ROFRec = o2::itsmft::ROFRecord;
  using MC2ROF = o2::itsmft::MC2ROFRecord;
  using HitVec = std::vector<Hit>;
  using MC2HITS_map = std::unordered_map<uint64_t, int>; // maps (track_ID<<16 + chip_ID) to entry in the hit vector

  std::vector<HitVec*> hitVecPool;
  std::vector<MC2HITS_map> mc2hitVec;

  TFile fout("CheckClusters.root", "recreate");
  TNtuple nt("ntc", "cluster ntuple", "chip:ev:lab:hlx:hlz:cgx:cgy:cgz:dx:dz");

  // Geometry
  o2::base::GeometryManager::loadGeometry(inputGeom);
  auto* gman = o2::iotof::GeometryTGeo::Instance();
  gman->fillMatrixCache(o2::math_utils::bit2Mask(o2::math_utils::TransformType::L2G));

  // Cluster topologies dictionary
  TFile* clsTopoFile = TFile::Open(topodictfile.data(), "READ");
  auto* clsTopoMapPtr = clsTopoFile->Get<std::unordered_map<uint32_t, o2::iotof::TopologyInfo>>("TF3ClusterTopologies");
  if (clsTopoMapPtr) {
    std::cout << "Loaded " << clsTopoMapPtr->size() << " entries from " << topodictfile << std::endl;
  } else {
    std::cerr << "Failed to load TF3ClusterTopologies from " << topodictfile << std::endl;
  }
  // Construct map directly from the vector pairs
  std::unordered_map<uint32_t, o2::iotof::TopologyInfo> topoMap(clsTopoMapPtr->begin(), clsTopoMapPtr->end());
  TopologyClassifier topoClassifier(std::move(topoMap));
  topoClassifier.setGeometry(gman);
  topoClassifier.print();
  clsTopoFile->Close();

  // Hits
  TFile fileH(hitfile.data());
  TTree* hitTree = (TTree*)fileH.Get("o2sim");
  std::vector<o2::itsmft::Hit>* hitArray = nullptr;
  hitTree->SetBranchAddress("TF3Hit", &hitArray);
  mc2hitVec.resize(hitTree->GetEntries());
  hitVecPool.resize(hitTree->GetEntries(), nullptr);
  int nEvts = hitTree->GetEntries();
  std::cout << "CheckClustersIOTOF: hitTree has " << hitTree->GetEntries() << " entries" << std::endl;

  // Clusters
  TFile fileC(clusfile.data());
  TTree* clusTree = (TTree*)fileC.Get("o2sim");
  clusTree->ls();
  std::vector<o2::iotof::Cluster>* clusArr = nullptr;
  clusTree->SetBranchAddress("TF3Cluster", &clusArr);
  std::vector<unsigned char>* patternsPtr = nullptr;
  auto pattBranch = clusTree->GetBranch("TF3ClusterPatt");
  if (pattBranch) {
    pattBranch->SetAddress(&patternsPtr);
  }
  std::cout << "CheckClustersIOTOF: clusTree has " << clusTree->GetEntries() << " entries" << std::endl;

  // ROFrecords
  std::vector<ROFRec> rofRecVec, *rofRecVecP = &rofRecVec;
  clusTree->SetBranchAddress("TF3ClusterROF", &rofRecVecP);
  std::cout << "CheckClustersIOTOF: rofRecVec has " << rofRecVec.size() << " entries" << std::endl;

  // Cluster MC labels
  o2::dataformats::MCTruthContainer<o2::MCCompLabel>* clusLabArr = nullptr;
  if (hitTree && clusTree->GetBranch("TF3ClusterMCTruth")) {
    clusTree->SetBranchAddress("TF3ClusterMCTruth", &clusLabArr);
  }

  clusTree->GetEntry(0);
  std::cout << "Number of clusters: " << clusArr->size() << std::endl;
  std::cout << "Number of pattern bytes: " << (patternsPtr ? patternsPtr->size() : 0) << std::endl;
  std::cout << "Number of label indices: " << (clusLabArr ? clusLabArr->getIndexedSize() : 0) << std::endl;
  // return;
  int nROFRec = (int)rofRecVec.size();

  // << build min and max MC events used by each ROF
  auto pattIt = patternsPtr->cbegin();
  int invalidPattIDCounter{0};
  for (int irof = 0; irof < nROFRec; irof++) {
    const auto& rofRec = rofRecVec[irof];
    rofRec.print();


    // >> read and map MC events contributing to this ROF
    for (int im = 0; im <= nEvts; im++) {
      if (!hitVecPool[im]) {
        hitTree->SetBranchAddress("TF3Hit", &hitVecPool[im]);
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
      int clEntry = icl; // entry of icl-th cluster of this ROF in the vector of clusters
      std::cout << "Processing cluster " << icl << "/" << rofRec.getNEntries() << std::endl;
      const auto& cluster = (*clusArr)[clEntry];

      float errX{0.f};
      float errZ{0.f};
      int npix = 0;
      uint16_t pattID = cluster.getPattern();
      uint8_t spanRow = cluster.getRowSpan();
      uint8_t spanCol = cluster.getColSpan();
      o2::math_utils::Point3D<float> locC;
      // std::cout << "CIAO1" << std::endl;
      if (pattID == o2::iotof::Cluster::InvalidPatternID) {
        invalidPattIDCounter++;
        continue;
      }
      // std::cout << "CIAO2" << std::endl;
      
      uint32_t topoKey = TopologyClassifier::makeKey(spanRow, spanCol, pattID);
      errX = topoClassifier.getErrX(topoKey);
      errZ = topoClassifier.getErrZ(topoKey);
      npix = topoClassifier.getNPixels(topoKey);
      auto chipID = cluster.getSensorID();
      // std::cout << "CIAO3" << std::endl;
      
      // Transformation to the local --> global
      locC = topoClassifier.getClusterCoordinates(cluster);
      // std::cout << "CIAO31" << std::endl;
      auto gloC = gman->getMatrixL2G(chipID) * locC;
      // std::cout << "CIAO32" << std::endl;
      
      // Check how many labels are there
      if (clusLabArr->getLabels(clEntry).empty()) {
        continue;
      }
      const auto& lab = (clusLabArr->getLabels(clEntry))[0];
      // std::cout << "CIAO33" << std::endl;
      
      // std::cout << "CIAO4" << std::endl;
      if (!lab.isValid() || lab.getSourceID() == QEDSourceID)
        continue;
      // std::cout << "CIAO5" << std::endl;
      
      // get MC info
      int trID = lab.getTrackID();
      const auto& mc2hit = mc2hitVec[lab.getEventID()];
      const auto* hitArray = hitVecPool[lab.getEventID()];
      uint64_t key = (uint64_t(trID) << 32) + chipID;
      auto hitEntry = mc2hit.find(key);
      if (hitEntry == mc2hit.end()) {
        LOG(error) << "Failed to find MC hit entry for Tr" << trID << " chipID" << chipID;
        continue;
      }
      // std::cout << "CIAO6" << std::endl;
      const auto& hit = (*hitArray)[hitEntry->second];
      //
      float dx = 0, dz = 0;
      int ievH = lab.getEventID();
      o2::math_utils::Point3D<float> locH, locHsta;
      
      // mean local position of the hit
      locH = gman->getMatrixL2G(chipID) ^ (hit.GetPos()); // inverse conversion from global to local
      locHsta = gman->getMatrixL2G(chipID) ^ (hit.GetPosStart());
      // std::cout << "CIAO7" << std::endl;
      auto x0 = locHsta.X(), dltx = locH.X() - x0;
      auto y0 = locHsta.Y(), dlty = locH.Y() - y0;
      auto z0 = locHsta.Z(), dltz = locH.Z() - z0;
      auto r = (0.5 * (chipInfo.SensorLayerThickness - chipInfo.SensorLayerThicknessEff) - y0) / dlty;
      locH.SetXYZ(x0 + r * dltx, y0 + r * dlty, z0 + r * dltz);
      // locH.SetXYZ(0.5 * (locH.X() + locHsta.X()), 0.5 * (locH.Y() + locHsta.Y()), 0.5 * (locH.Z() + locHsta.Z()));
      std::array<float, 10> data = {(float)chipID, (float)lab.getEventID(), (float)trID,
                                    locH.X(), locH.Z(),
                                    gloC.X(), gloC.Y(), gloC.Z(),
                                    locC.X() - locH.X(), locC.Z() - locH.Z()};
      // std::cout << "CIAO8" << std::endl;
      nt.Fill(data.data());
    }
  }
  std::cout << "CheckClustersIOTOF: Found " << invalidPattIDCounter << " clusters with invalid pattern ID" << std::endl;

  // distributions of differences between local positions of digits and hits in x and z
  auto canvdXdZ = new TCanvas("canvdXdZ", "", 1600, 800);
  canvdXdZ->Divide(2, 1);
  canvdXdZ->cd(1);
  nt.Draw("dx:dz>>h_dx_vs_dz_ITOF(600, -0.03, 0.03, 600, -0.03, 0.03)", "chip >= 0 && chip < 1920", "colz");
  addTLines(0.01);
  auto h = (TH2F*)gPad->GetPrimitive("h_dx_vs_dz_ITOF");
  Info("ITOF", "RMS(dx)=%.1f mu", h->GetRMS(2) * 1e4);
  Info("ITOF", "RMS(dz)=%.1f mu", h->GetRMS(1) * 1e4);
  canvdXdZ->cd(2);
  nt.Draw("dx:dz>>h_dx_vs_dz_OTOF(600, -0.03, 0.03, 600, -0.03, 0.03)", "chip >= 1920 && chip < 55488", "colz");
  addTLines(0.01);
  h = (TH2F*)gPad->GetPrimitive("h_dx_vs_dz_OTOF");
  Info("OTOF", "RMS(dx)=%.1f mu", h->GetRMS(2) * 1e4);
  Info("OTOF", "RMS(dz)=%.1f mu", h->GetRMS(1) * 1e4);
  canvdXdZ->SaveAs("tf3clusters_dx_vs_dz.pdf");
  canvdXdZ->SaveAs("tf3clusters_dx_vs_dz.root");

  fout.cd();
  nt.Write();
}
