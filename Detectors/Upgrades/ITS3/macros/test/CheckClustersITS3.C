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
/// \brief Simple macro to check ITSU clusters

#if !defined(__CLING__) || defined(__ROOTCLING__)
#include <TCanvas.h>
#include <TFile.h>
#include <TH2F.h>
#include <TNtuple.h>
#include <TString.h>
#include <TTree.h>
#include <TROOT.h>
#include <TStyle.h>

#define ENABLE_UPGRADES
#include "DetectorsCommonDataFormats/DetID.h"
#include "ITSMFTBase/SegmentationAlpide.h"
#include "ITS3Base/SegmentationSuperAlpide.h"
#include "ITS3Base/SpecsV2.h"
#include "ITSBase/GeometryTGeo.h"
#include "DataFormatsITSMFT/CompCluster.h"
#include "ITS3Reconstruction/TopologyDictionary.h"
#include "ITSMFTSimulation/Hit.h"
#include "DataFormatsITSMFT/ROFRecord.h"
#include "MathUtils/Cartesian.h"
#include "MathUtils/Utils.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "DetectorsCommonDataFormats/DetectorNameConf.h"
#endif

void CheckClustersITS3(std::string clusfile = "o2clus_its.root", std::string hitfile = "o2sim_HitsIT3.root",
                       std::string inputGeom = "o2sim_geometry.root", std::string dictfile = "ccdb/IT3/Calib/ClusterDictionary/snapshot.root", bool batch = false)
{
  gROOT->SetBatch(batch);

  using namespace o2::base;
  using namespace o2::its;

  using SuperSegmentation = o2::its3::SegmentationSuperAlpide;
  using Segmentation = o2::itsmft::SegmentationAlpide;
  using o2::itsmft::CompClusterExt;
  using o2::itsmft::Hit;
  using ROFRec = o2::itsmft::ROFRecord;
  using MC2ROF = o2::itsmft::MC2ROFRecord;
  using HitVec = std::vector<Hit>;
  using MC2HITS_map = std::unordered_map<uint64_t, int>; // maps (track_ID<<16 + chip_ID) to entry in the hit vector

  std::vector<HitVec*> hitVecPool;
  std::vector<MC2HITS_map> mc2hitVec;

  ULong_t cPattValid{0}, cPattInvalid{0};

  TFile fout("CheckClusters.root", "recreate");
  TNtuple nt("ntc", "cluster ntuple", "ev:lab:hlx:hlz:hgx:hgz:tx:tz:cgx:cgy:cgz:clx:cly:clz:dx:dy:dz:ex:ez:patid:rof:npx:id");

  // Geometry
  o2::base::GeometryManager::loadGeometry(inputGeom);
  auto gman = o2::its::GeometryTGeo::Instance();
  gman->fillMatrixCache(o2::math_utils::bit2Mask(o2::math_utils::TransformType::T2L, o2::math_utils::TransformType::T2GRot,
                                                 o2::math_utils::TransformType::L2G)); // request cached transforms

  // Hits
  TFile fileH(hitfile.data());
  auto* hitTree = dynamic_cast<TTree*>(fileH.Get("o2sim"));
  std::vector<o2::itsmft::Hit>* hitArray = nullptr;
  hitTree->SetBranchAddress("IT3Hit", &hitArray);
  mc2hitVec.resize(hitTree->GetEntries());
  hitVecPool.resize(hitTree->GetEntries(), nullptr);

  // Clusters
  TFile fileC(clusfile.data());
  auto* clusTree = dynamic_cast<TTree*>(fileC.Get("o2sim"));
  std::vector<CompClusterExt>* clusArr = nullptr;
  clusTree->SetBranchAddress("ITSClusterComp", &clusArr);
  std::vector<unsigned char>* patternsPtr = nullptr;
  auto pattBranch = clusTree->GetBranch("ITSClusterPatt");
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
  clusTree->SetBranchAddress("ITSClustersROF", &rofRecVecP);

  // Cluster MC labels
  o2::dataformats::MCTruthContainer<o2::MCCompLabel>* clusLabArr = nullptr;
  std::vector<MC2ROF> mc2rofVec, *mc2rofVecP = &mc2rofVec;
  if ((hitTree != nullptr) && (clusTree->GetBranch("ITSClusterMCTruth") != nullptr)) {
    clusTree->SetBranchAddress("ITSClusterMCTruth", &clusLabArr);
    clusTree->SetBranchAddress("ITSClustersMC2ROF", &mc2rofVecP);
  }

  clusTree->GetEntry(0);
  unsigned int nROFRec = (int)rofRecVec.size();
  std::vector<int> mcEvMin(nROFRec, hitTree->GetEntries());
  std::vector<int> mcEvMax(nROFRec, -1);

  // >> build min and max MC events used by each ROF
  for (int imc = mc2rofVec.size(); imc--;) {
    const auto& mc2rof = mc2rofVec[imc];
    // printf("MCRecord: ");
    // mc2rof.print();
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

    LOGP(debug, "Caching MC events contributing to this ROF {}", irof);
    for (int icl = 0; icl < rofRec.getNEntries(); icl++) {
      int clEntry = rofRec.getFirstEntry() + icl; // entry of icl-th cluster of this ROF in the vector of clusters
      const auto& cluster = (*clusArr)[clEntry];
      float errX{0.f};
      float errZ{0.f};
      int npix = 0;
      auto pattID = cluster.getPatternID();
      o2::math_utils::Point3D<float> locC;
      auto chipID = cluster.getSensorID();
      auto isIB = o2::its3::constants::detID::isDetITS3(chipID);
      auto layer = o2::its3::constants::detID::getDetID2Layer(chipID);
      auto clusterSize{-1};
      if (pattID == o2::itsmft::CompCluster::InvalidPatternID || dict.isGroup(pattID)) {
        o2::itsmft::ClusterPattern patt(pattIt);
        locC = dict.getClusterCoordinates(cluster, patt, false);
        LOGP(debug, "I am invalid and I am on chip {}", chipID);
        ++cPattInvalid;
        continue;
      } else {
        locC = dict.getClusterCoordinates(cluster);
        errX = dict.getErrX(pattID);
        errZ = dict.getErrZ(pattID);
        errX *= (isIB) ? SuperSegmentation::mPitchRow : Segmentation::PitchRow;
        errZ *= (isIB) ? SuperSegmentation::mPitchCol : Segmentation::PitchCol;
        npix = dict.getNpixels(pattID);
        ++cPattValid;
      }

      // Transformation to the local --> global
      auto gloC = gman->getMatrixL2G(chipID)(locC);
      const auto& lab = (clusLabArr->getLabels(clEntry))[0];

      if (!lab.isValid()) {
        continue;
      }

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
      const auto& hit = (*hitArray)[hitEntry->second];
      //
      float dx = 0, dz = 0;
      int ievH = lab.getEventID();
      o2::math_utils::Point3D<float> locH, locHsta;

      auto gloH = hit.GetPos();
      const auto& gloHsta = hit.GetPosStart();
      gloH.SetXYZ(0.5f * (gloH.X() + gloHsta.X()), 0.5f * (gloH.Y() + gloHsta.Y()), 0.5f * (gloH.Z() + gloHsta.Z()));

      // mean local position of the hit
      locH = gman->getMatrixL2G(chipID) ^ (hit.GetPos()); // inverse conversion from global to local
      locHsta = gman->getMatrixL2G(chipID) ^ (hit.GetPosStart());

      float x0 = locHsta.X(), dltx = locH.X() - x0;
      float y0 = locHsta.Y(), dlty = locH.Y() - y0;
      float z0 = locHsta.Z(), dltz = locH.Z() - z0;
      auto r = (0.5 * (Segmentation::SensorLayerThickness - Segmentation::SensorLayerThicknessEff) - y0) / dlty;

      if (!isIB) {
        locH.SetXYZ(x0 + r * dltx, y0 + r * dlty, z0 + r * dltz);
      } else {
        // compare in local flat coordinates
        float xFlatEnd{0.}, yFlatEnd{0.};
        o2::its3::SuperSegmentations[layer].curvedToFlat(locH.X(), locH.Y(), xFlatEnd, yFlatEnd);
        locH.SetXYZ(xFlatEnd, yFlatEnd, locH.Z());
        float xFlatSta{0.}, yFlatSta{0.};
        o2::its3::SuperSegmentations[layer].curvedToFlat(locHsta.X(), locHsta.Y(), xFlatSta, yFlatSta);
        locHsta.SetXYZ(xFlatSta, yFlatSta, locHsta.Z());
        // recalculate x/y in flat
        // x0 = xFlatSta, dltx = xFlatEnd - x0;
        // y0 = yFlatSta, dlty = yFlatEnd - y0;
        // r = (0.5 * (SuperSegmentation::mSensorLayerThickness - SuperSegmentation::mSensorLayerThicknessEff) - y0) / dlty;
        // locH.SetXYZ(x0 + r * dltx, y0 + r * dlty, z0 + r * dltz);

        // not really precise, but okish
        locH.SetXYZ(0.5f * (locH.X() + locHsta.X()), 0.5f * (locH.Y() + locHsta.Y()), 0.5f * (locH.Z() + locHsta.Z()));

        o2::its3::SuperSegmentations[layer].curvedToFlat(locC.X(), locC.Y(), xFlatSta, yFlatSta);
        locC.SetXYZ(xFlatSta, yFlatSta, locC.Z());
      }
      LOGP(debug, "{:*^30}", "START");
      LOGP(debug, "Cluster: X={:.5f} ; Z={:.5f} ; NPix={}", locC.X(), locC.Z(), npix);
      LOGP(debug, "Hit    : X={:.5f} ; Z={:.5f}", locH.X(), locH.Z());
      LOGP(debug, "{:*^30}", "END");

      std::array<float, 23> data = {(float)lab.getEventID(), (float)trID,
                                    locH.X(), locH.Z(),
                                    gloH.X(), gloH.Z(),
                                    dltx / dlty, dltz / dlty,
                                    gloC.X(), gloC.Y(), gloC.Z(),
                                    locC.X(), locC.Y(), locC.Z(),
                                    locC.X() - locH.X(), locC.Y() - locH.Y(), locC.Z() - locH.Z(),
                                    errX, errZ, (float)pattID,
                                    (float)rofRec.getROFrame(), (float)npix, (float)chipID};
      nt.Fill(data.data());
    }
  }

  LOGP(info, "There were {} valid PatternIDs and {} ({:.1f}%) invalid ones", cPattValid, cPattInvalid, ((float)cPattInvalid / (float)(cPattInvalid + cPattValid)) * 100);

  auto canvCgXCgY = new TCanvas("canvCgXCgY", "", 1600, 1600);
  canvCgXCgY->Divide(2, 2);
  canvCgXCgY->cd(1);
  nt.Draw("cgy:cgx>>h_cgy_vs_cgx_IB(1000, -5, 5, 1000, -5, 5)", "id < 3456", "colz");
  canvCgXCgY->cd(2);
  nt.Draw("cgy:cgz>>h_cgy_vs_cgz_IB(1000, -15, 15, 1000, -5, 5)", "id < 3456", "colz");
  canvCgXCgY->cd(3);
  nt.Draw("cgy:cgx>>h_cgy_vs_cgx_OB(1000, -50, 50, 1000, -50, 50)", "id >= 3456", "colz");
  canvCgXCgY->cd(4);
  nt.Draw("cgy:cgz>>h_cgy_vs_cgz_OB(1000, -100, 100, 1000, -50, 50)", "id >= 3456", "colz");
  canvCgXCgY->SaveAs("it3clusters_y_vs_x_vs_z.pdf");

  auto canvdXdZ = new TCanvas("canvdXdZ", "", 1600, 800);
  canvdXdZ->Divide(2, 2);
  canvdXdZ->cd(1)->SetLogz();
  nt.Draw("dx:dz>>h_dx_vs_dz_IB(1000, -0.01, 0.01, 1000, -0.01, 0.01)", "id < 3456", "colz");
  canvdXdZ->cd(2)->SetLogz();
  nt.Draw("dx:dz>>h_dx_vs_dz_OB(1000, -0.01, 0.01, 1000, -0.01, 0.01)", "id >= 3456", "colz");
  canvdXdZ->cd(3)->SetLogz();
  nt.Draw("dx:dz>>h_dx_vs_dz_IB_z(1000, -0.01, 0.01, 1000, -0.01, 0.01)", "id < 3456 && abs(cgz) < 2", "colz");
  canvdXdZ->cd(4)->SetLogz();
  nt.Draw("dx:dz>>h_dx_vs_dz_OB_z(1000, -0.01, 0.01, 1000, -0.01, 0.01)", "id >= 3456 && abs(cgz) < 2", "colz");
  canvdXdZ->SaveAs("it3clusters_dx_vs_dz.pdf");

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
