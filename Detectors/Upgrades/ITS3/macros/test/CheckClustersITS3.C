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
#include "ITS3Base/SegmentationMosaix.h"
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
#include "Steer/MCKinematicsReader.h"
#include "DetectorsCommonDataFormats/DetectorNameConf.h"
#endif

void CheckClustersITS3(const std::string& clusfile = "o2clus_its.root",
                       const std::string& hitfile = "o2sim_HitsIT3.root",
                       const std::string& inputGeom = "",
                       std::string dictfile = "../ccdb/IT3/Calib/ClusterDictionary/snapshot.root",
                       bool batch = true)
{
  gROOT->SetBatch(batch);

  using namespace o2::base;
  using namespace o2::its;

  using MosaixSegmentation = o2::its3::SegmentationMosaix;
  using Segmentation = o2::itsmft::SegmentationAlpide;
  using o2::itsmft::CompClusterExt;
  using o2::itsmft::Hit;
  using ROFRec = o2::itsmft::ROFRecord;
  using MC2ROF = o2::itsmft::MC2ROFRecord;
  using HitVec = std::vector<Hit>;
  using MC2HITS_map = std::unordered_map<uint64_t, std::vector<int>>; // maps (track_ID<<32 + chip_ID) to entry in the hit vector
  std::array<MosaixSegmentation, 3> mMosaixSegmentations{0, 1, 2};

  ULong_t cClustersIB{0}, cClustersOB{0};
  ULong_t cPattValidIB{0}, cPattInvalidIB{0}, cLabelInvalidIB{0}, cNoMCIB{0};
  ULong_t cPattValidOB{0}, cPattInvalidOB{0}, cLabelInvalidOB{0}, cNoMCOB{0};

  TFile fout("CheckClusters.root", "recreate");
  TNtuple nt("ntc", "cluster ntuple", "ev:lab:hlx:hlz:hgx:hgz:tx:tz:cgx:cgy:cgz:clx:cly:clz:dx:dy:dz:ex:ez:patid:rof:npx:id:eta:row:col:lay:prim");

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
  std::vector<HitVec*> hitVecPool;
  std::vector<MC2HITS_map> mc2hitVec;
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
  dict.print();

  //
  o2::steer::MCKinematicsReader reader("collisioncontext.root");

  // ROFrecords
  std::vector<ROFRec> rofRecVec, *rofRecVecP = &rofRecVec;
  clusTree->SetBranchAddress("ITSClustersROF", &rofRecVecP);

  // Cluster MC labels
  o2::dataformats::MCTruthContainer<o2::MCCompLabel>* clusLabArr = nullptr;
  std::vector<MC2ROF> mc2rofVec, *mc2rofVecP = &mc2rofVec;
  clusTree->SetBranchAddress("ITSClusterMCTruth", &clusLabArr);
  clusTree->GetEntry(0);
  auto pattIt = patternsPtr->cbegin();
  unsigned int nROFRec = (int)rofRecVec.size();

  // >> load all MC hits upfront
  for (int im = 0; im < (int)hitTree->GetEntries(); im++) {
    if (hitVecPool[im] == nullptr) {
      hitTree->SetBranchAddress("IT3Hit", &hitVecPool[im]);
      if (hitTree->GetEntry(im) <= 0 || hitVecPool[im] == nullptr) {
        LOGP(error, "Cannot read IT3Hit entry {} from {}", im, hitfile);
        return;
      }
      auto& mc2hit = mc2hitVec[im];
      const auto* hv = hitVecPool[im];
      for (int ih = (int)hv->size(); ih--;) {
        const auto& hit = (*hv)[ih];
        uint64_t key = (uint64_t(hit.GetTrackID()) << 32) + hit.GetDetectorID();
        mc2hit[key].push_back(ih);
      }
    }
  }

  for (unsigned int irof = 0; irof < nROFRec; irof++) {
    const auto& rofRec = rofRecVec[irof];
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
      (isIB) ? ++cClustersIB : ++cClustersOB;
      auto layer = gman->getLayer(chipID);
      auto clusterSize{-1};
      if (pattID == o2::itsmft::CompCluster::InvalidPatternID || dict.isGroup(pattID, isIB)) {
        o2::itsmft::ClusterPattern patt(pattIt);
        locC = dict.getClusterCoordinates(cluster, patt, false);
        LOGP(debug, "I am invalid and I am on chip {}", chipID);
        (isIB) ? ++cPattInvalidIB : ++cPattInvalidOB;
        continue;
      } else {
        locC = dict.getClusterCoordinates(cluster);
        errX = dict.getErrX(pattID, isIB);
        errZ = dict.getErrZ(pattID, isIB);
        npix = dict.getNpixels(pattID, isIB);
        (isIB) ? ++cPattValidIB : ++cPattValidOB;
      }

      // Transformation to the local --> global
      auto gloC = gman->getMatrixL2G(chipID)(locC);
      const auto& lab = (clusLabArr->getLabels(clEntry))[0];

      if (!lab.isValid()) {
        (isIB) ? ++cLabelInvalidIB : ++cLabelInvalidOB;
        continue;
      }
      auto track = reader.getTrack(lab);
      if (!track) {
        continue;
      }
      bool isPrimary = track->isPrimary();

      // get MC info
      const int trID = lab.getTrackID();
      const int evID = lab.getEventID();
      if (evID < 0 || evID >= (int)mc2hitVec.size()) {
        continue;
      }
      const auto& mc2hit = mc2hitVec[lab.getEventID()];
      uint64_t key = (uint64_t(trID) << 32) + chipID;
      auto hitEntry = mc2hit.find(key);
      if (hitEntry == mc2hit.end()) {
        LOG(debug) << "Failed to find MC hit entry for Tr" << trID << " chipID" << chipID;
        (isIB) ? ++cNoMCIB : ++cNoMCOB;
        continue;
      }

      const o2::itsmft::Hit* hit = nullptr;
      int nCand{0};
      for (const auto ih : hitEntry->second) {
        const auto& candHit = (*hitVecPool[evID])[ih];
        const int lay = gman->getLayer(candHit.GetDetectorID());
        if (layer == lay) {
          hit = &candHit;
          ++nCand;
        }
      }
      if (hit == nullptr || nCand > 1) {
        continue;
      }

      //
      float dx = 0, dz = 0;
      int ievH = lab.getEventID();
      o2::math_utils::Point3D<float> locH, locHsta;

      auto gloH = hit->GetPos();
      const auto& gloHsta = hit->GetPosStart();
      gloH.SetXYZ((gloH.X() + gloHsta.X()) * 0.5f, 0.5f * (gloH.Y() + gloHsta.Y()), 0.5f * (gloH.Z() + gloHsta.Z()));

      // mean local position of the hit
      locH = gman->getMatrixL2G(chipID) ^ (hit->GetPos()); // inverse conversion from global to local
      locHsta = gman->getMatrixL2G(chipID) ^ (hit->GetPosStart());

      float x0, y0, z0, dltx, dlty, dltz, r;
      if (isIB) {
        float xFlat{0.}, yFlat{0.};
        mMosaixSegmentations[layer].curvedToFlat(locC.X(), locC.Y(), xFlat, yFlat);
        locC.SetCoordinates(xFlat, yFlat, locC.Z());
        mMosaixSegmentations[layer].curvedToFlat(locH.X(), locH.Y(), xFlat, yFlat);
        locH.SetCoordinates(xFlat, yFlat, locH.Z());
        mMosaixSegmentations[layer].curvedToFlat(locHsta.X(), locHsta.Y(), xFlat, yFlat);
        locHsta.SetCoordinates(xFlat, yFlat, locHsta.Z());
        x0 = locHsta.X();
        dltx = locH.X() - x0;
        y0 = locHsta.Y();
        dlty = locH.Y() - y0;
        z0 = locHsta.Z();
        dltz = locH.Z() - z0;
        r = (o2::its3::constants::pixelarray::pixels::apts::responseYShift - y0) / dlty;
      } else {
        x0 = locHsta.X();
        dltx = locH.X() - x0;
        y0 = locHsta.Y();
        dlty = locH.Y() - y0;
        z0 = locHsta.Z();
        dltz = locH.Z() - z0;
        r = (0.5 * (Segmentation::SensorLayerThickness - Segmentation::SensorLayerThicknessEff) - y0) / dlty;
      }
      locH.SetXYZ(x0 + (r * dltx), y0 + (r * dlty), z0 + (r * dltz));

      float theta = std::acos(gloC.Z() / gloC.Rho());
      float eta = -std::log(std::tan(theta / 2));

      std::array<float, 28> data = {(float)lab.getEventID(), (float)trID,
                                    locH.X(), locH.Z(),
                                    gloH.X(), gloH.Z(),
                                    dltx / dlty, dltz / dlty,
                                    gloC.X(), gloC.Y(), gloC.Z(),
                                    locC.X(), locC.Y(), locC.Z(),
                                    locC.X() - locH.X(), locC.Y() - locH.Y(), locC.Z() - locH.Z(),
                                    errX, errZ, (float)pattID,
                                    (float)rofRec.getROFrame(), (float)npix, (float)chipID, eta, (float)cluster.getRow(), (float)cluster.getCol(), (float)layer, (float)isPrimary};
      nt.Fill(data.data());
    }
  }

  LOGP(info, "IB {} clusters and OB {} clusters", cClustersIB, cClustersOB);
  LOGP(info, "IB {} valid PatternIDs and {} ({:.1f}%) invalid ones", cPattValidIB, cPattInvalidIB, ((float)cPattInvalidIB / (float)(cPattInvalidIB + cPattValidIB)) * 100);
  LOGP(info, "IB {} invalid Labels and {} with No MC Hit information ", cLabelInvalidIB, cNoMCIB);
  LOGP(info, "OB {} valid PatternIDs and {} ({:.1f}%) invalid ones", cPattValidOB, cPattInvalidOB, ((float)cPattInvalidOB / (float)(cPattInvalidOB + cPattValidOB)) * 100);
  LOGP(info, "OB {} invalid Labels and {} with No MC Hit information ", cLabelInvalidOB, cNoMCOB);
  fout.cd();
  nt.Write();
}
