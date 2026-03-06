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
#include <unordered_map>

#include "DataFormatsTRK/Cluster.h"
#include "DataFormatsTRK/ROFRecord.h"
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

  using o2::MCCompLabel;
  using ROFRec = o2::trk::ROFRecord;
  using MC2ROF = o2::trk::MC2ROFRecord;
  using HitVec = std::vector<o2::trk::Hit>;
  using MC2HITS_map = std::unordered_map<uint64_t, int>; // maps (trackID << 32) + chipID -> hit index

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

  std::vector<o2::trk::Cluster>* clusArr = nullptr;
  std::vector<o2::trk::ROFRecord>* rofRecVecP = nullptr;
  std::vector<unsigned char>* patternsPtr = nullptr;
  clusTree->SetBranchAddress("TRKClusterComp", &clusArr);
  clusTree->SetBranchAddress("TRKClustersROF", &rofRecVecP);
  if (clusTree->GetBranch("TRKClusterPatt") != nullptr) {
    clusTree->SetBranchAddress("TRKClusterPatt", &patternsPtr);
  }

  o2::dataformats::MCTruthContainer<o2::MCCompLabel>* clusLabArr = nullptr;
  std::vector<MC2ROF> mc2rofVec, *mc2rofVecP = &mc2rofVec;
  bool hasMC = (clusTree->GetBranch("TRKClusterMCTruth") != nullptr);
  if (hasMC) {
    clusTree->SetBranchAddress("TRKClusterMCTruth", &clusLabArr);
    clusTree->SetBranchAddress("TRKClustersMC2ROF", &mc2rofVecP);
  }

  clusTree->GetEntry(0);
  const unsigned int nROFRec = rofRecVecP ? (unsigned int)rofRecVecP->size() : 0u;
  LOGP(info, "Number of ROF records: {}", nROFRec);
  auto pattIt = patternsPtr ? patternsPtr->cbegin() : std::vector<unsigned char>::const_iterator{};

  // ── Build per-ROF MC event range ───────────────────────────────────────────
  std::vector<int> mcEvMin(nROFRec, (int)hitTree->GetEntries());
  std::vector<int> mcEvMax(nROFRec, -1);
  if (hasMC) {
    for (int imc = (int)mc2rofVec.size(); imc--;) {
      const auto& mc2rof = mc2rofVec[imc];
      if (mc2rof.rofRecordID < 0) {
        continue;
      }
      for (unsigned int irfd = mc2rof.maxROF - mc2rof.minROF + 1; irfd--;) {
        unsigned int irof = mc2rof.rofRecordID + irfd;
        if (irof >= nROFRec) {
          continue;
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

  // ── Output ─────────────────────────────────────────────────────────────────
  TFile fout("CheckClusters.root", "recreate");
  // columns: event, MC track label,
  //   local hit x/z (flat frame), global hit x/y/z (midpoint),
  //   global cluster x/y/z, local cluster x/z,
  //   residuals dx/dz (local, cluster - hit),
  //   ROF frame, cluster size, chipID, layer, disk, subDetID, row, col, pt
  TNtuple nt("ntc", "TRK cluster ntuple",
             "event:mcTrackID:hitLocX:hitLocZ:hitGlobX:hitGlobY:hitGlobZ:clusGlobX:clusGlobY:clusGlobZ:clusLocX:clusLocZ:rofFrame:clusSize:chipID:layer:disk:subdet:row:col:pt");

  // ── Counters ───────────────────────────────────────────────────────────────
  long nTot{0}, nInvalidLabel{0}, nNoMCHit{0}, nValid{0};

  // ── Main loop ──────────────────────────────────────────────────────────────
  for (unsigned int irof = 0; irof < nROFRec; irof++) {
    const auto& rofRec = (*rofRecVecP)[irof];

    // Cache MC hit events for this ROF
    if (hasMC) {
      for (int im = mcEvMin[irof]; im <= mcEvMax[irof]; im++) {
        if (hitVecPool[im] == nullptr) {
          hitTree->SetBranchAddress("TRKHit", &hitVecPool[im]);
          hitTree->GetEntry(im);
          auto& mc2hit = mc2hitVec[im];
          const auto* hv = hitVecPool[im];
          for (int ih = (int)hv->size(); ih--;) {
            const auto& hit = (*hv)[ih];
            uint64_t key = (uint64_t(hit.GetTrackID()) << 32) + hit.GetDetectorID();
            mc2hit.emplace(key, ih);
          }
        }
      }
    }

    for (int icl = 0; icl < rofRec.getNEntries(); icl++) {
      const int clEntry = rofRec.getFirstEntry() + icl;
      const auto& cluster = (*clusArr)[clEntry];
      nTot++;

      // ── Parse pattern → center-of-gravity within bounding box ──────────
      // The cluster stores the bounding-box top-left pixel (row, col).
      // The pattern stream encodes [rowSpan, colSpan, bitmap...] for each cluster.
      // We accumulate pixel row/col offsets to obtain a sub-pixel CoG correction.
      float cogDr{0.f}, cogDc{0.f}; // mean offsets from bbox origin (pixels)
      if (patternsPtr) {
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
      // Formula from detectorToLocalUnchecked:
      //   VD  : xRow = 0.5*(width[lay]-pitchRow) - row*pitchRow  → row↑ xRow↓
      //         zCol = col*pitchCol + 0.5*(pitchCol-length)      → col↑ zCol↑
      //   MLOT: same structure with MLOT pitches
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
      clLocX -= cogDr * pitchRow; // increasing row → decreasing xRow
      clLocZ += cogDc * pitchCol; // increasing col → increasing zCol
      const float yResponse = (cluster.subDetID == 0) ? yPlaneVD : yPlaneMLOT;
      // For VD the L2G matrix is built in the *curved* local frame (quasi-Cartesian,
      // origin at the beam axis). Convert flat (clLocX, 0) → curved (xC, yC) first.
      // For MLOT (flat sensors) the local frame is already Cartesian: pass directly.
      // clLocX is already in the flat frame from detectorToLocalUnchecked + CoG and
      // does NOT need any further transformation for the residual comparison.
      o2::math_utils::Point3D<float> locC;
      if (cluster.subDetID == 0) {
        auto cv = o2::trk::SegmentationChip::flatToCurved(cluster.layer, clLocX, 0.f);
        locC = {cv.X(), cv.Y(), clLocZ};
      } else {
        locC = {clLocX, yResponse, clLocZ};
      }
      auto gloC = gman->getMatrixL2G(cluster.chipID)(locC);

      if (!hasMC || clusLabArr == nullptr) {
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

      // ── Find matching MC hit ────────────────────────────────────────────
      const auto& mc2hit = mc2hitVec[evID];
      uint64_t key = (uint64_t(trID) << 32) + cluster.chipID;
      auto hitEntry = mc2hit.find(key);
      if (hitEntry == mc2hit.end()) {
        nNoMCHit++;
        continue;
      }
      const auto& hit = (*hitVecPool[evID])[hitEntry->second];
      const float pt = TMath::Hypot(hit.GetPx(), hit.GetPy());

      // ── Hit global midpoint ────────────────────────────────────────────
      const auto& gloHend = hit.GetPos();
      const auto& gloHsta = hit.GetPosStart();
      o2::math_utils::Point3D<float> gloHmid(
        0.5f * (gloHend.X() + gloHsta.X()),
        0.5f * (gloHend.Y() + gloHsta.Y()),
        0.5f * (gloHend.Z() + gloHsta.Z()));

      // ── Hit global → local ─────────────────────────────
      o2::math_utils::Point3D<float> locHsta = gman->getMatrixL2G(cluster.chipID) ^ (gloHsta); // inverse L2G
      o2::math_utils::Point3D<float> locHend = gman->getMatrixL2G(cluster.chipID) ^ (gloHend); // inverse L2G

      // ── Propagate hit segment to the sensor response surface ───────────────
      // Rather than the geometric midpoint, find where the track segment crosses
      // the response plane (y = responseYShift in the flat local frame).
      // For VD (curved): convert both endpoints to flat frame first.
      // For ML/OT (flat): use local coordinates directly.
      float hitLocX{0.f}, hitLocZ{0.f};
      if (cluster.subDetID == 0) { // VD – curved sensor
        auto flatSta = o2::trk::SegmentationChip::curvedToFlat(cluster.layer, locHsta.X(), locHsta.Y());
        auto flatEnd = o2::trk::SegmentationChip::curvedToFlat(cluster.layer, locHend.X(), locHend.Y());
        float x0 = flatSta.X(), y0 = flatSta.Y(), z0 = locHsta.Z();
        float dltx = flatEnd.X() - x0, dlty = flatEnd.Y() - y0, dltz = locHend.Z() - z0;
        float r = (std::abs(dlty) > 1e-9f) ? (yPlaneVD - y0) / dlty : 0.5f;
        hitLocX = x0 + r * dltx;
        hitLocZ = z0 + r * dltz;
      } else { // ML/OT – flat sensor
        float x0 = locHsta.X(), y0 = locHsta.Y(), z0 = locHsta.Z();
        float dltx = locHend.X() - x0, dlty = locHend.Y() - y0, dltz = locHend.Z() - z0;
        float r = (std::abs(dlty) > 1e-9f) ? (yPlaneMLOT - y0) / dlty : 0.5f;
        hitLocX = x0 + r * dltx;
        hitLocZ = z0 + r * dltz;
      }

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

  // ── Summary ────────────────────────────────────────────────────────────────
  LOGP(info, "=== TRK Cluster vs Hit Summary ===");
  LOGP(info, "Total clusters:          {}", nTot);
  LOGP(info, "Valid (hit matched):     {}", nValid);
  LOGP(info, "Invalid/noise MC labels: {}", nInvalidLabel);
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
  nt.Draw("hitLocX-clusLocX>>h_dx_VD(200,-0.02,0.02)", "subdet==0&&event>=0");
  canvRes->cd(2)->SetLogy();
  nt.Draw("hitLocZ-clusLocZ>>h_dz_VD(200,-0.02,0.02)", "subdet==0&&event>=0");
  canvRes->cd(3)->SetLogy();
  nt.Draw("hitLocX-clusLocX>>h_dx_MLOT(200,-0.02,0.02)", "subdet==1&&event>=0");
  canvRes->cd(4)->SetLogy();
  nt.Draw("hitLocZ-clusLocZ>>h_dz_MLOT(200,-0.02,0.02)", "subdet==1&&event>=0");
  canvRes->cd(5)->SetLogz();
  nt.Draw("hitLocX-clusLocX:hitLocZ-clusLocZ>>h_dxdz_VD(200,-0.02,0.02,200,-0.02,0.02)", "subdet==0&&event>=0", "colz");
  canvRes->cd(6);
  nt.Draw("hitLocX-clusLocX:hitLocZ-clusLocZ>>h_dxdz_MLOT(200,-0.02,0.02,200,-0.02,0.02)", "subdet==1&&event>=0", "colz");
  canvRes->SaveAs("trk_residuals.png");

  auto canvResVsLayer = new TCanvas("canvResVsLayer", "Residuals vs layer", 1600, 600);
  canvResVsLayer->Divide(2, 1);
  canvResVsLayer->cd(1);
  nt.Draw("hitLocX-clusLocX:layer>>h_dx_vs_lay(20,0,20,200,-0.02,0.02)", "event>=0", "prof");
  canvResVsLayer->cd(2);
  nt.Draw("hitLocZ-clusLocZ:layer>>h_dz_vs_lay(20,0,20,200,-0.02,0.02)", "event>=0", "prof");
  canvResVsLayer->SaveAs("trk_residuals_vs_layer.png");

  fout.cd();
  nt.Write();
  fout.Close();

  LOGP(info, "Output saved to CheckClusters.root and PNG files");
}
