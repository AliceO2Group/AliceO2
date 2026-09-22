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

/// \file CheckClustersIOTOF.C
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

void addTLines(float pitchRow, float pitchCol)
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
  float xRow = 0.f;
  while (xRow > xmin) {
    TLine* lineNeg = new TLine(xRow, ymin, xRow, ymax);
    lineNeg->SetLineStyle(2);
    lineNeg->SetLineColor(kGray + 3);
    lineNeg->Draw("same");
    TLine* linePos = new TLine(std::abs(xRow), ymin, std::abs(xRow), ymax);
    linePos->SetLineStyle(2);
    linePos->SetLineColor(kGray + 3);
    linePos->Draw("same");
    xRow -= pitchRow / 2;
  }

  float yCol = 0.f;
  while (yCol > ymin) {
    TLine* lineNeg = new TLine(xmin, yCol, xmax, yCol);
    lineNeg->SetLineStyle(2);
    lineNeg->SetLineColor(kGray + 3);
    lineNeg->Draw("same");
    TLine* linePos = new TLine(xmin, std::abs(yCol), xmax, std::abs(yCol));
    linePos->SetLineStyle(2);
    linePos->SetLineColor(kGray + 3);
    linePos->Draw("same");
    yCol -= pitchCol / 2;
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
  // trackID + chipID --> eventID + hitIndex
  using MC2HITS_map = std::unordered_map<uint64_t, std::vector<int>>; // maps (track_ID<<16 + chip_ID) to entry in the hit vector

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
  // for (int irof = 0; irof < nROFRec; irof++) {
  const auto& rofRec = rofRecVec[0];
  rofRec.print();

  // >> read and map MC events contributing to this ROF
  // for (int im = 0; im <= nEvts; im++) {
  for (int im = 0; im < nEvts; im++) {
    if (!hitVecPool[im]) {
      hitTree->SetBranchAddress("TF3Hit", &hitVecPool[im]);
      hitTree->GetEntry(im);
      auto& mc2hit = mc2hitVec[im];
      const auto* hitArray = hitVecPool[im];
      for (int ih = hitArray->size(); ih--;) {
        const auto& hit = (*hitArray)[ih];
        uint64_t key = (uint64_t(hit.GetTrackID()) << 32) + hit.GetDetectorID();
        mc2hit[key].push_back(ih);
      }
    }
  }

  // << cache MC events contributing to this ROF
  for (int clEntry = 0; clEntry < rofRec.getNEntries(); clEntry++) {
    std::cout << "\nProcessing cluster " << clEntry << "/" << rofRec.getNEntries() << std::endl;
    const auto& cluster = (*clusArr)[clEntry];

    uint16_t pattID = cluster.getPattern();
    o2::math_utils::Point3D<float> locC;
    if (pattID == o2::iotof::Cluster::InvalidPatternID) {
      invalidPattIDCounter++;
      continue;
    }

    auto chipID = cluster.getSensorID();

    // Transformation to the local --> global
    locC = topoClassifier.getClusterCoordinates(cluster);
    auto gloC = gman->getMatrixL2G(chipID) * locC;

    // Check how many labels are there
    if (clusLabArr->getLabels(clEntry).empty()) {
      continue;
    }
    const auto& lab = (clusLabArr->getLabels(clEntry))[0];

    if (!lab.isValid() || lab.getSourceID() == QEDSourceID)
      continue;

    // get MC info
    int trID = lab.getTrackID();
    int evID = lab.getEventID();
    const auto& mc2hit = mc2hitVec[lab.getEventID()];
    const auto* hitArray = hitVecPool[lab.getEventID()];
    uint64_t key = (uint64_t(trID) << 32) + chipID;
    auto hitEntry = mc2hit.find(key);
    if (hitEntry == mc2hit.end()) {
      LOG(error) << "Failed to find MC hit entry for Track: " << trID << ", chipID: " << chipID;
      continue;
    }

    if (hitEntry->second.size() == 0) {
      LOG(error) << "No hits found for Track: " << trID << ", chipID: " << chipID;
      continue;
    }
    o2::math_utils::Point3D<float> locH, locHsta;
    int closestHitIdx = -1;
    if (hitEntry->second.size() == 1) {
      closestHitIdx = 0;
    } else {
      float maxDist = std::numeric_limits<float>::max();
      for (int iHitIdx = 0; iHitIdx < hitEntry->second.size(); iHitIdx++) {
        const o2::itsmft::Hit* hit = &((*hitArray)[hitEntry->second[iHitIdx]]);
        if (!hit) {
          LOG(error) << "Failed to find matching hit for Track: " << trID << ", chipID: " << chipID << ", eventID: " << evID;
          continue;
        }
        locH = gman->getMatrixL2G(chipID) ^ (hit->GetPos()); // inverse conversion from global to local
        locHsta = gman->getMatrixL2G(chipID) ^ (hit->GetPosStart());
        locH.SetXYZ(0.5 * (locH.X() + locHsta.X()), 0.5 * (locH.Y() + locHsta.Y()), 0.5 * (locH.Z() + locHsta.Z()));
        float dx = std::abs(locC.X() - locH.X());
        float dz = std::abs(locC.Z() - locH.Z());
        float dist = std::sqrt(dx * dx + dz * dz);
        if (maxDist > dist) {
          maxDist = dist;
          closestHitIdx = iHitIdx;
        }
      }
    }
    const o2::itsmft::Hit* hit = &((*hitArray)[hitEntry->second[closestHitIdx]]);
    if (!hit) {
      LOG(error) << "Failed to find matching hit for cluster " << clEntry << std::endl;
      continue;
    }
    locH = gman->getMatrixL2G(chipID) ^ (hit->GetPos()); // inverse conversion from global to local
    locHsta = gman->getMatrixL2G(chipID) ^ (hit->GetPosStart());
    locH.SetXYZ(0.5 * (locH.X() + locHsta.X()), 0.5 * (locH.Y() + locHsta.Y()), 0.5 * (locH.Z() + locHsta.Z()));

    // mean local position of the hit
    std::array<float, 10> data = {(float)chipID, (float)lab.getEventID(), (float)trID,
                                  locH.X(), locH.Z(),
                                  gloC.X(), gloC.Y(), gloC.Z(),
                                  locC.X() - locH.X(), locC.Z() - locH.Z()};
    nt.Fill(data.data());
  }
  // } ROF loop
  std::cout << "CheckClustersIOTOF: Found " << invalidPattIDCounter << " clusters with invalid pattern ID" << std::endl;

  // cluster maps in the xy and yz planes
  auto canvXY = new TCanvas("canvXY", "", 1600, 800);
  canvXY->Divide(2, 1);
  canvXY->cd(1);
  nt.Draw("cgy:cgx>>h_y_vs_x_IOTOF(1000, -100, 100, 1000, -100, 100)", "chip >= 0 && chip < 55488", "colz");
  canvXY->cd(2);
  nt.Draw("cgy:cgz>>h_y_vs_z_IOTOF(1000, -400, 400, 1000, -100, 100)", "chip >= 0 && chip < 55488", "colz");
  canvXY->SaveAs("tf3clusters_y_vs_x_vs_z.pdf");
  canvXY->SaveAs("tf3clusters_y_vs_x_vs_z.root");

  // distributions of differences between local positions of digits and hits in x and z
  float canvaEdgeRow = 1.25 * chipInfo.PitchRow;
  float canvaEdgeCol = 1.25 * chipInfo.PitchCol;
  auto canvdXdZ = new TCanvas("canvdXdZ", "", 1600, 800);
  canvdXdZ->Divide(2, 1);
  canvdXdZ->cd(1);
  nt.Draw(Form("dx:dz>>h_dx_vs_dz_ITOF(600, -%f, %f, 600, -%f, %f)", canvaEdgeRow, canvaEdgeRow, canvaEdgeCol, canvaEdgeCol), "chip >= 0 && chip < 1920", "colz");
  addTLines(chipInfo.PitchRow, chipInfo.PitchCol);
  auto h = (TH2F*)gPad->GetPrimitive("h_dx_vs_dz_ITOF");
  Info("ITOF", "RMS(dx)=%.1f mu", h->GetRMS(2) * 1e4);
  Info("ITOF", "RMS(dz)=%.1f mu", h->GetRMS(1) * 1e4);
  canvdXdZ->cd(2);
  nt.Draw(Form("dx:dz>>h_dx_vs_dz_OTOF(600, -%f, %f, 600, -%f, %f)", canvaEdgeRow, canvaEdgeRow, canvaEdgeCol, canvaEdgeCol), "chip >= 1920 && chip < 55488", "colz");
  addTLines(chipInfo.PitchRow, chipInfo.PitchCol);
  h = (TH2F*)gPad->GetPrimitive("h_dx_vs_dz_OTOF");
  Info("OTOF", "RMS(dx)=%.1f mu", h->GetRMS(2) * 1e4);
  Info("OTOF", "RMS(dz)=%.1f mu", h->GetRMS(1) * 1e4);
  canvdXdZ->SaveAs("tf3clusters_dx_vs_dz.pdf");
  canvdXdZ->SaveAs("tf3clusters_dx_vs_dz.root");

  fout.cd();
  nt.Write();
}
