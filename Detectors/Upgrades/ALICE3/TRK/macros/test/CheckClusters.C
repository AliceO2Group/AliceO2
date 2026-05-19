// Copyright 2019-2026 CERN and copyright holders of ALICE O2.
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
/// \brief Macro to check TRK clusters and compare cluster positions to MC hit positions

#if !defined(__CLING__) || defined(__ROOTCLING__)
#include <TCanvas.h>
#include <TFile.h>
#include <TH1F.h>
#include <TH2F.h>
#include <TNtuple.h>
#include <TString.h>
#include <TTree.h>
#include <TROOT.h>
#include <TStyle.h>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <unordered_map>
#include <vector>

#include "DataFormatsTRK/Cluster.h"
#include "DataFormatsTRK/ROFRecord.h"
#include "TRKBase/AlmiraParam.h"
#include "TRKBase/GeometryTGeo.h"
#include "TRKBase/SegmentationChip.h"
#include "TRKSimulation/Hit.h"
#include "ITSMFTSimulation/AlpideSimResponse.h"
#include "CCDB/BasicCCDBManager.h"
#include "MathUtils/Cartesian.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "DetectorsBase/GeometryManager.h"
#include "Framework/Logger.h"
#endif

void CheckClusters(const std::string& clusfile = "o2clus_trk.root",
                   const std::string& hitfile = "o2sim_HitsTRK.root",
                   const std::string& inputGeom = "o2sim_geometry.root",
                   const std::string& ccdbUrl = "http://alice-ccdb.cern.ch",
                   long ccdbTimestamp = -1,
                   bool batch = false)
{
  gROOT->SetBatch(batch);

  using HitVec = std::vector<o2::trk::Hit>;
  using MC2HITS_map = std::unordered_map<uint64_t, std::vector<int>>; // maps (trackID << 32) + chipID -> hit indices

  // ── Chip response (for hit-segment propagation to charge-collection plane) ──
  // Fetches the same AlpideSimResponse from CCDB as the digitizer (IT3/Calib/APTSResponse)
  // and computes Y-intersection planes with the same formulas from Digitizer::init()
  auto& ccdbMgr = o2::ccdb::BasicCCDBManager::instance();
  ccdbMgr.setURL(ccdbUrl);
  if (ccdbTimestamp > 0) {
    ccdbMgr.setTimestamp(ccdbTimestamp);
  }
  auto* alpResp = ccdbMgr.get<o2::itsmft::AlpideSimResponse>("IT3/Calib/APTSResponse");
  if (!alpResp) {
    LOGP(fatal, "Cannot retrieve AlpideSimResponse from CCDB at {}", ccdbUrl);
    return;
  }
  const float depthMax = alpResp->getDepthMax();

  // ── Y-plane shifts: why VD and ML/OT need different values ────────────────
  //
  // The APTS pixel response (AlpideSimResponse) uses an internal Y axis where:
  //
  //        y = depthMax  ──  beam-entry (top) surface
  //        y = 0         ──  charge-collection plane   ← where clusters form
  //        y < 0         ──  substrate (no response)
  //
  // The digitizer (Digitizer::init()) brings hit Y coordinates into this frame
  // by adding a per-sub-detector shift before querying the response:
  //
  //     y_APTS = y_local + shift                  [Digitizer.cxx ::processHit]
  //
  // The collection plane (y_APTS = 0) is therefore at  y_local = −shift
  // in the detector local frame.  That is the Y value used here when
  // propagating the MC hit segment to a single representative point.
  //
  // ── VD (vertex detector – curved sensors) ─────────────────────────────────
  // After SegmentationChip::curvedToFlat() (convention: yFlat = dist − R):
  //   outer face (beam-entry): yFlat = +halfThickVD = +10 µm
  //   inner face (exit):       yFlat = −halfThickVD = −10 µm
  // The digitizer uses:
  //
  //     mSimRespVDShift = depthMax − halfThickVD
  //
  // so the collection plane (y_APTS = 0) corresponds to:
  //
  //     yPlaneVD = alice3resp::responseYShift = +5 µm
  //
  // i.e. 5 µm inside from the outer (entry) face. ✓
  //
  // ── ML/OT (middle/outer tracker – flat sensors) ────────────────────────────
  // The local Y origin is at the GEOMETRIC CENTRE of the sensor volume.
  // The outer (entry) surface is at y_local = +SiliconThicknessMLOT/2.
  // The digitizer uses:
  //
  //     mSimRespMLOTShift = depthMax − SiliconThicknessMLOT / 2
  //
  // so the collection plane (y_APTS = 0) is at:
  //
  //     yPlaneMLOT = SiliconThicknessMLOT/2 − depthMax
  //
  // ──────────────────────────────────────────────────────────────────────────
  const float halfThicknessMLOT = o2::trk::SegmentationChip::SiliconThicknessMLOT / 2.f;
  const float yPlaneVD = (float)o2::trk::constants::alice3resp::responseYShift; // VD: collection plane 5 µm inside outer (entry) face in flat local frame
  const float yPlaneMLOT = halfThicknessMLOT - depthMax;                        // MLOT: entry @ +halfThick, collection depthMax below entry
  LOGP(info, "Response depthMax = {:.4f} cm  |  VD Y-plane = {:.4f} cm  |  ML/OT Y-plane = {:.4f} cm",
       depthMax, yPlaneVD, yPlaneMLOT);

  // ── Geometry ───────────────────────────────────────────────────────────────
  o2::base::GeometryManager::loadGeometry(inputGeom);
  auto gman = o2::trk::GeometryTGeo::Instance();
  gman->fillMatrixCache(o2::math_utils::bit2Mask(o2::math_utils::TransformType::L2G));

  // ── Hits ───────────────────────────────────────────────────────────────────
  TFile fileH(hitfile.data());
  auto* hitTree = dynamic_cast<TTree*>(fileH.Get("o2sim"));
  if (!hitTree) {
    LOGP(error, "Cannot find o2sim tree in {}", hitfile);
    return;
  }
  if (hitTree->GetBranch("TRKHit") == nullptr) {
    LOGP(error, "Cannot find TRKHit branch in {}", hitfile);
    return;
  }
  std::vector<MC2HITS_map> mc2hitVec;
  std::vector<HitVec*> hitVecPool;
  mc2hitVec.resize(hitTree->GetEntries());
  hitVecPool.resize(hitTree->GetEntries(), nullptr);

  // ── Clusters ───────────────────────────────────────────────────────────────
  TFile fileC(clusfile.data());
  auto* clusTree = dynamic_cast<TTree*>(fileC.Get("o2sim"));
  if (!clusTree) {
    LOGP(error, "Cannot find o2sim tree in {}", clusfile);
    return;
  }

  // Read per-layer cluster branches and accumulate
  static constexpr int nLayers = o2::trk::AlmiraParam::kNLayers;
  std::vector<std::vector<o2::trk::Cluster>*> clusArrPerLayer(nLayers, nullptr);
  std::vector<std::vector<o2::trk::ROFRecord>*> rofRecVecPerLayer(nLayers, nullptr);
  std::vector<std::vector<unsigned char>*> patternsPerLayer(nLayers, nullptr);
  std::vector<o2::dataformats::MCTruthContainer<o2::MCCompLabel>*> clusLabArrPerLayer(nLayers, nullptr);
  std::vector<std::vector<size_t>> patternOffsetsPerLayer(nLayers);
  std::vector<bool> layerActive(nLayers, false);

  bool hasAnyMC = false;
  for (int iLayer = 0; iLayer < nLayers; iLayer++) {
    std::string brClus = std::string("TRKClusterComp_") + std::to_string(iLayer);
    std::string brROF = std::string("TRKClustersROF_") + std::to_string(iLayer);
    std::string brPatt = std::string("TRKClusterPatt_") + std::to_string(iLayer);
    std::string brMCTruth = std::string("TRKClusterMCTruth_") + std::to_string(iLayer);

    if (clusTree->GetBranch(brClus.c_str()) == nullptr) {
      LOGP(warning, "Branch {} not found, skipping layer {}", brClus, iLayer);
      continue;
    }
    if (clusTree->GetBranch(brROF.c_str()) == nullptr) {
      LOGP(error, "Branch {} not found, skipping layer {}", brROF, iLayer);
      continue;
    }
    clusTree->SetBranchAddress(brClus.c_str(), &clusArrPerLayer[iLayer]);
    clusTree->SetBranchAddress(brROF.c_str(), &rofRecVecPerLayer[iLayer]);
    if (clusTree->GetBranch(brPatt.c_str()) != nullptr) {
      clusTree->SetBranchAddress(brPatt.c_str(), &patternsPerLayer[iLayer]);
    } else {
      LOGP(warning, "Branch {} not found, layer {} cluster positions use bbox origins", brPatt, iLayer);
    }
    if (clusTree->GetBranch(brMCTruth.c_str()) != nullptr) {
      clusTree->SetBranchAddress(brMCTruth.c_str(), &clusLabArrPerLayer[iLayer]);
      hasAnyMC = true;
    }
    layerActive[iLayer] = true;
  }

  // Read entry and accumulate all layers
  if (clusTree->GetEntry(0) <= 0) {
    LOGP(error, "Cannot read entry 0 from {}", clusfile);
    return;
  }

  auto hasAnyActiveLayer = false;
  for (int iLayer = 0; iLayer < nLayers; iLayer++) {
    hasAnyActiveLayer = hasAnyActiveLayer || layerActive[iLayer];
  }
  if (!hasAnyActiveLayer) {
    LOGP(error, "No usable TRK cluster layers found in {}", clusfile);
    return;
  }

  // Print total clusters per layer
  for (int iLayer = 0; iLayer < nLayers; iLayer++) {
    if (!layerActive[iLayer]) {
      continue;
    }
    if (clusArrPerLayer[iLayer] == nullptr || rofRecVecPerLayer[iLayer] == nullptr) {
      LOGP(error, "Layer {} branches were declared but did not load usable data, skipping layer", iLayer);
      layerActive[iLayer] = false;
      continue;
    }
    LOGP(info, "Layer {}: {} clusters", iLayer, clusArrPerLayer[iLayer]->size());
  }

  // The pattern stream is variable-length, so index it by cluster entry once.
  for (int iLayer = 0; iLayer < nLayers; iLayer++) {
    if (!layerActive[iLayer] || patternsPerLayer[iLayer] == nullptr) {
      continue;
    }
    const auto nClusters = clusArrPerLayer[iLayer]->size();
    const auto& patterns = *patternsPerLayer[iLayer];
    auto& offsets = patternOffsetsPerLayer[iLayer];
    offsets.resize(nClusters, std::numeric_limits<size_t>::max());
    size_t pattPos = 0;
    bool validPatterns = true;
    for (size_t icl = 0; icl < nClusters; icl++) {
      if (pattPos + 2 > patterns.size()) {
        validPatterns = false;
        break;
      }
      offsets[icl] = pattPos;
      const uint8_t rowSpan = patterns[pattPos];
      const uint8_t colSpan = patterns[pattPos + 1];
      const size_t nBytes = (size_t(rowSpan) * colSpan + 7) / 8;
      if (pattPos + 2 + nBytes > patterns.size()) {
        validPatterns = false;
        break;
      }
      pattPos += 2 + nBytes;
    }
    if (!validPatterns || pattPos != patterns.size()) {
      LOGP(error, "Malformed pattern stream for layer {}: {} pattern bytes for {} clusters, disabling CoG corrections for this layer",
           iLayer, patterns.size(), nClusters);
      patternsPerLayer[iLayer] = nullptr;
      offsets.clear();
    }
  }

  // Accumulate max ROF count across all layers
  unsigned int nROFRec = 0;
  for (int iLayer = 0; iLayer < nLayers; iLayer++) {
    if (!layerActive[iLayer]) {
      continue;
    }
    nROFRec = std::max(nROFRec, (unsigned int)rofRecVecPerLayer[iLayer]->size());
  }
  LOGP(info, "Number of ROF records: {}", nROFRec);

  // ── Load all MC hit events upfront (TRK has no MC2ROF mapping) ──────────────
  if (hasAnyMC) {
    LOGP(info, "Pre-loading {} MC events", hitTree->GetEntries());
    for (int im = 0; im < (int)hitTree->GetEntries(); im++) {
      if (hitVecPool[im] == nullptr) {
        hitTree->SetBranchAddress("TRKHit", &hitVecPool[im]);
        if (hitTree->GetEntry(im) <= 0 || hitVecPool[im] == nullptr) {
          LOGP(error, "Cannot read TRKHit entry {} from {}", im, hitfile);
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
  }

  // ── Output ─────────────────────────────────────────────────────────────────
  TFile fout("CheckClusters.root", "recreate");
  // columns: event, MC track label,
  //   local hit x/z (flat frame), global hit x/y/z (midpoint),
  //   global cluster x/y/z, local cluster x/z,
  //   ROF frame, cluster size, chipID, layer, disk, subDetID, row, col, pt
  TNtuple nt("ntc", "TRK cluster ntuple",
             "event:mcTrackID:hitLocX:hitLocZ:hitGlobX:hitGlobY:hitGlobZ:clusGlobX:clusGlobY:clusGlobZ:clusLocX:clusLocZ:rofFrame:clusSize:chipID:layer:disk:subdet:row:col:pt");

  // ── Counters ───────────────────────────────────────────────────────────────
  long nTot{0}, nInvalidLabel{0}, nInvalidEvent{0}, nNoMCHit{0}, nValid{0};

  // ── Main loop ──────────────────────────────────────────────────────────────
  for (unsigned int irof = 0; irof < nROFRec; irof++) {
    // Process each layer
    for (int iLayer = 0; iLayer < nLayers; iLayer++) {
      if (!layerActive[iLayer]) {
        continue;
      }
      if (rofRecVecPerLayer[iLayer]->empty() || irof >= rofRecVecPerLayer[iLayer]->size()) {
        continue;
      }
      const auto& rofRec = (*rofRecVecPerLayer[iLayer])[irof];
      const auto& clusArr = *clusArrPerLayer[iLayer];
      const auto& clusLabArr = clusLabArrPerLayer[iLayer];
      const auto* patternsPtr = patternsPerLayer[iLayer];
      const auto& patternOffsets = patternOffsetsPerLayer[iLayer];

      for (int icl = 0; icl < rofRec.getNEntries(); icl++) {
        const int clEntry = rofRec.getFirstEntry() + icl;
        if (clEntry < 0 || clEntry >= (int)clusArr.size()) {
          LOGP(error, "Layer {} ROF {} points to cluster entry {} outside {} clusters",
               iLayer, irof, clEntry, clusArr.size());
          continue;
        }
        const auto& cluster = clusArr[clEntry];
        nTot++;

        // ── Parse pattern → center-of-gravity within bounding box ──────────
        // Keep this in sync with Clusterer::getClusterLocalCoordinates().
        float cogDr{0.f}, cogDc{0.f}; // mean offsets from bbox origin (pixels)
        if (patternsPtr != nullptr) {
          const auto pattOffset = patternOffsets[clEntry];
          const auto* pattIt = patternsPtr->data() + pattOffset;
          const uint8_t rowSpan = *pattIt++;
          const uint8_t colSpan = *pattIt++;
          const int nBytes = (rowSpan * colSpan + 7) / 8;
          int nPix{0}, pixIdx{0};
          for (int ib = 0; ib < nBytes; ib++) {
            const uint8_t byte = *pattIt++;
            for (int bit = 7; bit >= 0 && pixIdx < rowSpan * colSpan; bit--, pixIdx++) {
              if (byte & (1 << bit)) {
                cogDr += pixIdx / colSpan;
                cogDc += pixIdx % colSpan;
                nPix++;
              }
            }
          }
          if (nPix > 1) {
            cogDr /= nPix;
            cogDc /= nPix;
          }
        }

        // ── Cluster local → global (CoG position) ─────────────────────────────
        // Get local coords of the bounding-box corner pixel, then apply the
        // fractional CoG displacement using the pixel pitch.
        float clLocX{0.f}, clLocZ{0.f};
        o2::trk::SegmentationChip::detectorToLocalUnchecked(
          cluster.row, cluster.col, clLocX, clLocZ,
          cluster.subDetID, cluster.layer, cluster.disk);
        const float pitchRow = (cluster.subDetID == 0)
                                 ? o2::trk::SegmentationChip::PitchRowVD
                                 : o2::trk::SegmentationChip::PitchRowMLOT;
        const float pitchCol = (cluster.subDetID == 0)
                                 ? o2::trk::SegmentationChip::PitchColVD
                                 : o2::trk::SegmentationChip::PitchColMLOT;
        clLocX -= cogDr * pitchRow; // increasing row -> decreasing xRow
        clLocZ += cogDc * pitchCol; // increasing col -> increasing zCol

        o2::math_utils::Point3D<float> locC;
        if (cluster.subDetID == 0) {
          auto cv = o2::trk::SegmentationChip::flatToCurved(cluster.layer, clLocX, 0.f);
          locC = {cv.X(), cv.Y(), clLocZ};
        } else {
          locC = {clLocX, yPlaneMLOT, clLocZ};
        }
        auto gloC = gman->getMatrixL2G(cluster.chipID)(locC);

        if (!hasAnyMC || clusLabArr == nullptr) {
          // No MC info: just fill geometry columns, leave residuals as 0
          std::array<float, 21> data = {
            -1.f, -1.f,
            0.f, 0.f, 0.f, 0.f, 0.f,
            (float)gloC.X(), (float)gloC.Y(), (float)gloC.Z(),
            clLocX, clLocZ,
            (float)rofRec.getROFrame(), (float)cluster.size, (float)cluster.chipID,
            (float)cluster.layer, (float)cluster.disk, (float)cluster.subDetID,
            (float)cluster.row, (float)cluster.col, -1.f};
          nt.Fill(data.data());
          continue;
        }

        // ── MC label ───────────────────────────────────────────────────────
        const auto& labels = clusLabArr->getLabels(clEntry);
        if (labels.empty() || !labels[0].isValid()) {
          nInvalidLabel++;
          continue;
        }
        const auto& lab = labels[0];
        const int trID = lab.getTrackID();
        const int evID = lab.getEventID();
        if (evID < 0 || evID >= (int)mc2hitVec.size()) {
          nInvalidEvent++;
          continue;
        }

        // ── Find matching MC hit ────────────────────────────────────────────
        const auto& mc2hit = mc2hitVec[evID];
        uint64_t key = (uint64_t(trID) << 32) + cluster.chipID;
        auto hitEntry = mc2hit.find(key);
        if (hitEntry == mc2hit.end()) {
          nNoMCHit++;
          continue;
        }
        auto projectHitToResponsePlane = [&](const o2::trk::Hit& hit, float& hitLocX, float& hitLocZ) {
          const auto& gloHend = hit.GetPos();
          const auto& gloHsta = hit.GetPosStart();
          o2::math_utils::Point3D<float> locHsta = gman->getMatrixL2G(cluster.chipID) ^ (gloHsta); // inverse L2G
          o2::math_utils::Point3D<float> locHend = gman->getMatrixL2G(cluster.chipID) ^ (gloHend); // inverse L2G

          // Rather than the geometric midpoint, find where the track segment crosses
          // the response plane. For VD convert the curved endpoints to the flat frame first.
          if (cluster.subDetID == 0) {
            auto flatSta = o2::trk::SegmentationChip::curvedToFlat(cluster.layer, locHsta.X(), locHsta.Y());
            auto flatEnd = o2::trk::SegmentationChip::curvedToFlat(cluster.layer, locHend.X(), locHend.Y());
            float x0 = flatSta.X(), y0 = flatSta.Y(), z0 = locHsta.Z();
            float dltx = flatEnd.X() - x0, dlty = flatEnd.Y() - y0, dltz = locHend.Z() - z0;
            float r = (std::abs(dlty) > 1e-9f) ? (yPlaneVD - y0) / dlty : 0.5f;
            hitLocX = x0 + r * dltx;
            hitLocZ = z0 + r * dltz;
          } else {
            float x0 = locHsta.X(), y0 = locHsta.Y(), z0 = locHsta.Z();
            float dltx = locHend.X() - x0, dlty = locHend.Y() - y0, dltz = locHend.Z() - z0;
            float r = (std::abs(dlty) > 1e-9f) ? (yPlaneMLOT - y0) / dlty : 0.5f;
            hitLocX = x0 + r * dltx;
            hitLocZ = z0 + r * dltz;
          }
        };

        const o2::trk::Hit* bestHit = nullptr;
        float hitLocX{0.f}, hitLocZ{0.f};
        float bestDist2 = std::numeric_limits<float>::max();
        for (const auto ih : hitEntry->second) {
          const auto& candHit = (*hitVecPool[evID])[ih];
          float candLocX{0.f}, candLocZ{0.f};
          projectHitToResponsePlane(candHit, candLocX, candLocZ);
          const float dx = clLocX - candLocX;
          const float dz = clLocZ - candLocZ;
          const float dist2 = dx * dx + dz * dz;
          if (dist2 < bestDist2) {
            bestDist2 = dist2;
            bestHit = &candHit;
            hitLocX = candLocX;
            hitLocZ = candLocZ;
          }
        }
        if (bestHit == nullptr) {
          nNoMCHit++;
          continue;
        }
        const auto& hit = *bestHit;
        const float pt = TMath::Hypot(hit.GetPx(), hit.GetPy());

        // ── Hit global midpoint ────────────────────────────────────────────
        const auto& gloHend = hit.GetPos();
        const auto& gloHsta = hit.GetPosStart();
        o2::math_utils::Point3D<float> gloHmid(
          0.5f * (gloHend.X() + gloHsta.X()),
          0.5f * (gloHend.Y() + gloHsta.Y()),
          0.5f * (gloHend.Z() + gloHsta.Z()));

        nValid++;
        std::array<float, 21> data = {
          (float)evID, (float)trID,
          hitLocX, hitLocZ,
          (float)gloHmid.X(), (float)gloHmid.Y(), (float)gloHmid.Z(),
          (float)gloC.X(), (float)gloC.Y(), (float)gloC.Z(),
          clLocX, clLocZ,
          (float)rofRec.getROFrame(), (float)cluster.size, (float)cluster.chipID,
          (float)cluster.layer, (float)cluster.disk, (float)cluster.subDetID,
          (float)cluster.row, (float)cluster.col, pt};
        nt.Fill(data.data());
      }
    }
  }

  // ── Summary ────────────────────────────────────────────────────────────────
  LOGP(info, "=== TRK Cluster vs Hit Summary ===");
  LOGP(info, "Total clusters:          {}", nTot);
  LOGP(info, "Valid (hit matched):     {}", nValid);
  LOGP(info, "Invalid/noise MC labels: {}", nInvalidLabel);
  LOGP(info, "Invalid MC event IDs:    {}", nInvalidEvent);
  LOGP(info, "MC hit not found:        {}", nNoMCHit);
  // ── Visualisation ──────────────────────────────────────────────────────────
  auto canvGlobal = new TCanvas("canvGlobal", "Cluster global positions", 1600, 800);
  canvGlobal->Divide(2, 1);
  canvGlobal->cd(1);
  nt.Draw("clusGlobY:clusGlobX>>h_yx(500,-50,50,500,-50,50)", "", "colz");
  canvGlobal->cd(2);
  nt.Draw("clusGlobY:clusGlobZ>>h_yz(500,-100,100,500,-50,50)", "", "colz");
  canvGlobal->SaveAs("trk_clusters_global.png");

  auto canvRes = new TCanvas("canvRes", "Residuals (cluster - hit) [cm]", 1600, 1200);
  canvRes->Divide(2, 3);
  canvRes->cd(1)->SetLogy();
  nt.Draw("clusLocX-hitLocX>>h_dx_VD(200,-0.02,0.02)", "subdet==0&&event>=0");
  canvRes->cd(2)->SetLogy();
  nt.Draw("clusLocZ-hitLocZ>>h_dz_VD(200,-0.02,0.02)", "subdet==0&&event>=0");
  canvRes->cd(3)->SetLogy();
  nt.Draw("clusLocX-hitLocX>>h_dx_MLOT(200,-0.02,0.02)", "subdet==1&&event>=0");
  canvRes->cd(4)->SetLogy();
  nt.Draw("clusLocZ-hitLocZ>>h_dz_MLOT(200,-0.02,0.02)", "subdet==1&&event>=0");
  canvRes->cd(5)->SetLogz();
  nt.Draw("clusLocX-hitLocX:clusLocZ-hitLocZ>>h_dxdz_VD(200,-0.02,0.02,200,-0.02,0.02)", "subdet==0&&event>=0", "colz");
  canvRes->cd(6);
  nt.Draw("clusLocX-hitLocX:clusLocZ-hitLocZ>>h_dxdz_MLOT(200,-0.02,0.02,200,-0.02,0.02)", "subdet==1&&event>=0", "colz");
  canvRes->SaveAs("trk_residuals.png");

  auto canvResVsLayer = new TCanvas("canvResVsLayer", "Residuals vs layer", 1600, 600);
  canvResVsLayer->Divide(2, 1);
  canvResVsLayer->cd(1);
  nt.Draw("clusLocX-hitLocX:layer>>h_dx_vs_lay(20,0,20,200,-0.02,0.02)", "event>=0", "prof");
  canvResVsLayer->cd(2);
  nt.Draw("clusLocZ-hitLocZ:layer>>h_dz_vs_lay(20,0,20,200,-0.02,0.02)", "event>=0", "prof");
  canvResVsLayer->SaveAs("trk_residuals_vs_layer.png");

  fout.cd();
  nt.Write();
  fout.Close();

  LOGP(info, "Output saved to CheckClusters.root and PNG files");
}
