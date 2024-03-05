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

/// \file CompareClustersAndDigits.C
/// \brief Simple macro to compare ITS3 clusters and digits

#if !defined(__CLING__) || defined(__ROOTCLING__)
#include <TCanvas.h>
#include <TFile.h>
#include <TGraph.h>
#include <TH2F.h>
#include <TLegend.h>
#include <TNtuple.h>
#include <TROOT.h>
#include <TString.h>
#include <TArrow.h>
#include <TStyle.h>
#include <TTree.h>

#define ENABLE_UPGRADES
#include "DataFormatsITSMFT/CompCluster.h"
#include "DataFormatsITSMFT/Digit.h"
#include "DataFormatsITSMFT/ROFRecord.h"
#include "DetectorsCommonDataFormats/DetID.h"
#include "DetectorsCommonDataFormats/DetectorNameConf.h"
#include "ITS3Base/SegmentationSuperAlpide.h"
#include "ITS3Base/SpecsV2.h"
#include "ITS3Reconstruction/TopologyDictionary.h"
#include "ITSBase/GeometryTGeo.h"
#include "ITSMFTBase/SegmentationAlpide.h"
#include "ITSMFTSimulation/Hit.h"
#include "MathUtils/Cartesian.h"
#include "MathUtils/Utils.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "SimulationDataFormat/ConstMCTruthContainer.h"
#include "SimulationDataFormat/IOMCTruthContainerView.h"

#include <filesystem>
#endif

struct Data {
  TH2F* pixelArray;
  TGraph* hitS;
  TGraph* hitM;
  TGraph* hitE;
  TGraph* clusters;
  TGraph* cog;
  TLegend* leg;
  std::vector<TBox*>* vBoxes;
  void clear()
  {
    delete pixelArray;
    delete hitS;
    delete hitM;
    delete hitE;
    delete clusters;
    delete cog;
    delete leg;
    for (auto& b : *vBoxes) {
      delete b;
    }
    delete vBoxes;
  }
};

void CompareClustersAndDigits(std::string clusfile = "o2clus_it3.root",
                              std::string digifile = "it3digits.root",
                              std::string dictfile = "IT3dictionary.root",
                              std::string hitfile = "o2sim_HitsIT3.root",
                              std::string inputGeom = "o2sim_geometry.root",
                              bool batch = true)
{
  TH1::AddDirectory(kFALSE);
  gROOT->SetBatch(batch);
  gStyle->SetPalette(kRainBow);
  gStyle->SetOptStat(0);

  using namespace o2::base;
  using o2::itsmft::Hit;
  using SuperSegmentation = o2::its3::SegmentationSuperAlpide;
  using Segmentation = o2::itsmft::SegmentationAlpide;
  using o2::itsmft::CompClusterExt;
  using ROFRec = o2::itsmft::ROFRecord;
  using MC2ROF = o2::itsmft::MC2ROFRecord;
  using HitVec = std::vector<Hit>;
  using MC2HITS_map = std::unordered_map<uint64_t, int>; // maps (track_ID<<16 + chip_ID) to
                                                         // entry in the hit vector
  std::vector<HitVec*> hitVecPool;
  std::vector<MC2HITS_map> mc2hitVec;

  // Geometry
  o2::base::GeometryManager::loadGeometry(inputGeom);
  auto gman = o2::its::GeometryTGeo::Instance();
  gman->fillMatrixCache(o2::math_utils::bit2Mask(o2::math_utils::TransformType::T2L, o2::math_utils::TransformType::T2GRot,
                                                 o2::math_utils::TransformType::L2G)); // request cached transforms
  const int nChips = gman->getNumberOfChips();

  // Hits
  TFile fileH(hitfile.data());
  auto* hitTree = dynamic_cast<TTree*>(fileH.Get("o2sim"));
  std::vector<o2::itsmft::Hit>* hitArray = nullptr;
  hitTree->SetBranchAddress("IT3Hit", &hitArray);
  mc2hitVec.resize(hitTree->GetEntries());
  hitVecPool.resize(hitTree->GetEntries(), nullptr);

  // Digits
  TFile* digFile = TFile::Open(digifile.data());
  TTree* digTree = (TTree*)digFile->Get("o2sim");
  std::vector<o2::itsmft::Digit>* digArr = nullptr;
  digTree->SetBranchAddress("IT3Digit", &digArr);
  o2::dataformats::IOMCTruthContainerView* plabels = nullptr;
  digTree->SetBranchAddress("IT3DigitMCTruth", &plabels);

  // Clusters
  TFile fileC(clusfile.data());
  auto* clusTree = dynamic_cast<TTree*>(fileC.Get("o2sim"));
  std::vector<CompClusterExt>* clusArr = nullptr;
  clusTree->SetBranchAddress("IT3ClusterComp", &clusArr);
  std::vector<unsigned char>* patternsPtr = nullptr;
  auto pattBranch = clusTree->GetBranch("IT3ClusterPatt");
  if (pattBranch != nullptr) {
    pattBranch->SetAddress(&patternsPtr);
  }
  if (dictfile.empty()) {
    dictfile = o2::base::DetectorNameConf::getAlpideClusterDictionaryFileName(o2::detectors::DetID::IT3, "", "root");
  }
  o2::its3::TopologyDictionary dict;
  std::ifstream file(dictfile.c_str());
  if (file.good()) {
    LOG(info) << "Running with dictionary: " << dictfile.c_str();
    dict.readFromFile(dictfile);
  } else {
    LOG(info) << "Running without dictionary !";
  }

  // ROFrecords
  std::vector<ROFRec> rofRecVec, *rofRecVecP = &rofRecVec;
  clusTree->SetBranchAddress("IT3ClustersROF", &rofRecVecP);

  // Cluster MC labels
  o2::dataformats::MCTruthContainer<o2::MCCompLabel>* clusLabArr = nullptr;
  std::vector<MC2ROF> mc2rofVec, *mc2rofVecP = &mc2rofVec;
  if ((hitTree != nullptr) && (clusTree->GetBranch("IT3ClusterMCTruth") != nullptr)) {
    clusTree->SetBranchAddress("IT3ClusterMCTruth", &clusLabArr);
    clusTree->SetBranchAddress("IT3ClustersMC2ROF", &mc2rofVecP);
  }

  clusTree->GetEntry(0);
  unsigned int nROFRec = (int)rofRecVec.size();
  std::vector<int> mcEvMin(nROFRec, hitTree->GetEntries());
  std::vector<int> mcEvMax(nROFRec, -1);
  /* digTree->GetEntry(0); */
  /* o2::dataformats::ConstMCTruthContainer<o2::MCCompLabel> labels; */
  /* plabels->copyandflatten(labels); */
  /* delete plabels; */

  // >> build min and max MC events used by each ROF
  for (int imc = mc2rofVec.size(); imc--;) {
    const auto& mc2rof = mc2rofVec[imc];
    if (mc2rof.rofRecordID < 0) {
      continue; // this MC event did not contribute to any ROF
    }
    for (unsigned int irfd = mc2rof.maxROF - mc2rof.minROF + 1; irfd--;) {
      unsigned int irof = mc2rof.rofRecordID + irfd;
      if (irof >= nROFRec) {
        LOG(error) << "ROF=" << irof << " from MC2ROF record is >= N ROFs=" << nROFRec;
      }
      if (mcEvMin[irof] > imc) {
        mcEvMin[irof] = imc;
      }
      if (mcEvMax[irof] < imc) {
        mcEvMax[irof] = imc;
      }
    }
  }

  // Create all plots
  LOGP(info, "Creating plots");
  std::vector<Data> data(nChips);
  for (int iChip{0}; iChip < nChips; ++iChip) {
    auto& dat = data[iChip];
    int col{o2::its3::SegmentationSuperAlpide::mNCols}, row{o2::its3::SegmentationSuperAlpide::mNRows};
    if (!o2::its3::constants::detID::isDetITS3(iChip)) {
      col = o2::itsmft::SegmentationAlpide::NCols;
      row = o2::itsmft::SegmentationAlpide::NRows;
    }
    dat.pixelArray = new TH2F(Form("histSensor_%d", iChip), Form("SensorID=%d;col;row", iChip), col, -0.5, col - 0.5, row, -0.5, row - 0.5);
    dat.hitS = new TGraph();
    dat.hitS->SetMarkerStyle(22);
    dat.hitS->SetMarkerColor(kGreen);
    dat.hitS->SetEditable(kFALSE);
    dat.hitM = new TGraph();
    dat.hitM->SetMarkerStyle(20);
    dat.hitM->SetMarkerColor(kGreen + 3);
    dat.hitM->SetEditable(kFALSE);
    dat.hitE = new TGraph();
    dat.hitE->SetMarkerStyle(23);
    dat.hitE->SetMarkerColor(kGreen + 5);
    dat.hitE->SetEditable(kFALSE);
    dat.clusters = new TGraph(1);
    dat.clusters->SetMarkerStyle(29);
    dat.clusters->SetMarkerColor(kBlue);
    dat.clusters->SetEditable(kFALSE);
    dat.cog = new TGraph(1);
    dat.cog->SetMarkerStyle(21);
    dat.cog->SetMarkerColor(kRed);
    dat.cog->SetEditable(kFALSE);
    dat.leg = new TLegend(0.7, 0.7, 0.92, 0.92);
    dat.leg->AddEntry(dat.clusters, "Cluster Start");
    dat.leg->AddEntry(dat.cog, "Cluster COG");
    dat.leg->AddEntry(dat.hitS, "Hit Start");
    dat.leg->AddEntry(dat.hitM, "Hit Middle");
    dat.leg->AddEntry(dat.hitE, "Hit End");
    dat.vBoxes = new std::vector<TBox*>;
  }

  LOGP(info, "Filling digits");
  for (int iDigit{0}; digTree->LoadTree(iDigit) >= 0; ++iDigit) {
    digTree->GetEntry(iDigit);
    for (const auto& digit : *digArr) {
      const auto chipID = digit.getChipIndex();
      data[chipID].pixelArray->Fill(digit.getColumn(), digit.getRow());
    }
  }

  LOGP(info, "Building min and max MC events used by each ROF");
  auto pattIt = patternsPtr->cbegin();
  for (unsigned int irof = 0; irof < nROFRec; irof++) {
    const auto& rofRec = rofRecVec[irof];
    // >> read and map MC events contributing to this ROF
    for (int im = mcEvMin[irof]; im <= mcEvMax[irof]; im++) {
      if (hitVecPool[im] == nullptr) {
        hitTree->SetBranchAddress("IT3Hit", &hitVecPool[im]);
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

    // Clusters in this ROF
    for (int icl = 0; icl < rofRec.getNEntries(); icl++) {
      int clEntry = rofRec.getFirstEntry() + icl; // entry of icl-th cluster of this ROF in the vector of clusters
      const auto& cluster = (*clusArr)[clEntry];
      const auto chipID = cluster.getSensorID();
      const auto pattID = cluster.getPatternID();
      const auto isIB = o2::its3::constants::detID::isDetITS3(chipID);
      const auto layer = gman->getLayer(chipID);
      if (pattID == o2::itsmft::CompCluster::InvalidPatternID || dict.isGroup(pattID)) {
        continue;
      }
      const auto& lab = (clusLabArr->getLabels(clEntry))[0];
      if (!lab.isValid()) {
        continue;
      }
      const int trID = lab.getTrackID();
      const auto& mc2hit = mc2hitVec[lab.getEventID()];
      const auto* hitArray = hitVecPool[lab.getEventID()];
      uint64_t key = (uint64_t(trID) << 32) + chipID;
      auto hitEntry = mc2hit.find(key);
      if (hitEntry == mc2hit.end()) {
        LOG(error) << "Failed to find MC hit entry for Tr" << trID << " chipID" << chipID;
        continue;
      }
      auto locC = dict.getClusterCoordinates(cluster);
      const auto& hit = (*hitArray)[hitEntry->second];
      auto locHEnd = gman->getMatrixL2G(chipID) ^ (hit.GetPos());
      auto locHStart = gman->getMatrixL2G(chipID) ^ (hit.GetPosStart());
      o2::math_utils::Point3D<float> locHMiddle;
      if (isIB) {
        float xFlat{0.}, yFlat{0.};
        o2::its3::SuperSegmentations[layer].curvedToFlat(locHEnd.X(), locHEnd.Y(), xFlat, yFlat);
        locHEnd.SetXYZ(xFlat, yFlat, locHEnd.Z());
        o2::its3::SuperSegmentations[layer].curvedToFlat(locHStart.X(), locHStart.Y(), xFlat, yFlat);
        locHStart.SetXYZ(xFlat, yFlat, locHStart.Z());
      }
      locHMiddle.SetXYZ(0.5f * (locHEnd.X() + locHStart.X()), 0.5f * (locHEnd.Y() + locHStart.Y()), 0.5f * (locHEnd.Z() + locHStart.Z()));

      int rowHS, colHS, rowHM, colHM, rowHE, colHE, colC, rowC;
      bool v1, v2, v3, v4;
      if (isIB) {
        v1 = o2::its3::SuperSegmentations[layer].localToDetector(locHStart.X(), locHStart.Z(), rowHS, colHS);
        v2 = o2::its3::SuperSegmentations[layer].localToDetector(locHMiddle.X(), locHMiddle.Z(), rowHM, colHM);
        v3 = o2::its3::SuperSegmentations[layer].localToDetector(locHEnd.X(), locHEnd.Z(), rowHE, colHE);
        v4 = o2::its3::SuperSegmentations[layer].localToDetector(locC.X(), locC.Z(), rowC, colC);
      } else {
        v1 = o2::itsmft::SegmentationAlpide::localToDetector(locHStart.X(), locHStart.Z(), rowHS, colHS);
        v2 = o2::itsmft::SegmentationAlpide::localToDetector(locHMiddle.X(), locHMiddle.Z(), rowHM, colHM);
        v3 = o2::itsmft::SegmentationAlpide::localToDetector(locHEnd.X(), locHEnd.Z(), rowHE, colHE);
        v4 = o2::itsmft::SegmentationAlpide::localToDetector(locC.X(), locC.Z(), rowC, colC);
      }
      if (!v1 || !v2 || !v3 || !v4) {
        // sometimes the transformation for hit start/end do not work since they can beyond the chip if they are
        // at the edge, so for visualisation purposes we do not draw these clusters
        continue;
      }

      data[chipID].hitS->AddPoint(colHS, rowHS);
      data[chipID].hitM->AddPoint(colHM, rowHM);
      data[chipID].hitE->AddPoint(colHE, rowHE);
      data[chipID].clusters->AddPoint(cluster.getCol(), cluster.getRow());
      data[chipID].cog->AddPoint(colC, rowC);

      constexpr float delta = 1e-2;
      const auto& patt = dict.getPattern(cluster.getPatternID());
      auto box = new TBox(
        cluster.getCol() - delta - 0.5,
        cluster.getRow() - delta - 0.5,
        cluster.getCol() + patt.getColumnSpan() + delta - 0.5,
        cluster.getRow() + patt.getRowSpan() + delta - 0.5);
      box->SetFillColorAlpha(0, 0);
      box->SetFillStyle(0);
      box->SetLineWidth(2);
      box->SetLineColor(kBlack);
      data[chipID].vBoxes->push_back(box);
    }
  }

  LOGP(info, "Writing to root file");
  double x1, y1, x2, y2;
  std::unique_ptr<TFile> oFile(TFile::Open("CompareClustersAndDigits.root", "RECREATE"));
  for (int iChip{0}; iChip < nChips; ++iChip) {
    if (iChip > 100) {
      break;
    }
    auto& dat = data[iChip];
    gFile->cd();
    /* auto path = gman->getMatrixPath(iChip); */
    TString path; // TODO wrong use above
    const std::string cpath{path.Data() + 39, path.Data() + path.Length()};
    const std::filesystem::path p{cpath};
    if (oFile->mkdir(p.parent_path().c_str(), "", true) == nullptr) {
      LOGP(error, "Cannot create directories with path: %s", p.parent_path().c_str());
      continue;
    }
    if (!gDirectory->cd(p.parent_path().c_str())) {
      LOGP(error, "Failed to cd %s", p.parent_path().c_str());
      continue;
    }
    auto canv = new TCanvas(Form("%s_%d", p.filename().c_str(), iChip));
    canv->SetTitle(Form("%s_%d", p.filename().c_str(), iChip));
    canv->cd();
    gPad->SetGrid(1, 1);
    dat.pixelArray->Draw("colz");
    dat.clusters->Draw("p;same");
    dat.cog->Draw("p;same");
    dat.hitS->Draw("p;same");
    dat.hitM->Draw("p;same");
    dat.hitE->Draw("p;same");
    for (const auto& b : *dat.vBoxes) {
      b->Draw();
    }
    auto arr = new TArrow();
    arr->SetArrowSize(0.02);
    for (int i{0}; i < dat.hitS->GetN(); ++i) {
      dat.hitS->GetPoint(i, x1, y1);
      dat.hitE->GetPoint(i, x2, y2);
      arr->DrawArrow(x1, y1, x2, y2);
    }

    dat.leg->Draw();
    canv->SetEditable(false);

    gDirectory->WriteTObject(canv);
    dat.clear();
    delete canv;
    delete arr;
    printf("\rWriting chip %05d / %d", iChip, nChips - 1);
  }
  printf("\r\n");
  LOGP(info, "Finished!");
}
