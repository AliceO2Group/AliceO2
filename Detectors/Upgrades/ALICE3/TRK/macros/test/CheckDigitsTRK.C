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

/// \file CheckDigits.C
/// \brief Simple macro to check TRK digits

#if !defined(__CLING__) || defined(__ROOTCLING__)
#include <TCanvas.h>
#include <TFile.h>
#include <TH2F.h>
#include <TNtuple.h>
#include <TString.h>
#include <TTree.h>
#include <TLine.h>
#include <TStyle.h>

#include "TRKBase/SegmentationChip.h"
#include "TRKBase/GeometryTGeo.h"
#include "DataFormatsITSMFT/Digit.h"
#include "TRKSimulation/Hit.h"
#include "MathUtils/Utils.h"
#include "SimulationDataFormat/ConstMCTruthContainer.h"
#include "SimulationDataFormat/IOMCTruthContainerView.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "DetectorsBase/GeometryManager.h"
#include "ITSMFTSimulation/AlpideSimResponse.h"
#include "CCDB/BasicCCDBManager.h"

#include "DataFormatsITSMFT/ROFRecord.h"

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

void CheckDigits(std::string digifile = "trkdigits.root", std::string hitfile = "o2sim_HitsTRK.root", std::string inputGeom = "o2sim_geometry.root")
{
  gStyle->SetPalette(55);

  using namespace o2::base;
  using namespace o2::trk;

  using o2::itsmft::Digit;
  using o2::trk::Hit;

  using o2::trk::SegmentationChip;

  TFile* f = TFile::Open("CheckDigits.root", "recreate");

  TNtuple* nt = new TNtuple("ntd", "digit ntuple", "id:x:y:z:rowD:colD:rowH:colH:xlH:zlH:xlcH:zlcH:dx:dz");
  TNtuple* nt2 = new TNtuple("ntd2", "digit ntuple", "id:z:dxH:dzH"); /// maximum number of elements in a tuple = 15: doing a new tuple to store more variables

  // Geometry
  o2::base::GeometryManager::loadGeometry(inputGeom);
  auto* gman = o2::trk::GeometryTGeo::Instance();
  gman->fillMatrixCache(o2::math_utils::bit2Mask(o2::math_utils::TransformType::L2G));

  const int nVDLayers = gman->extractNumberOfLayersVD();
  const int nMLOTLayers = gman->getNumberOfLayersMLOT();
  const int nTotalLayers = nVDLayers + nMLOTLayers;

  SegmentationChip seg;
  // seg.Print();

  // MLOT response plane: y = halfThickness - depthMax.
  float depthMax = (float)o2::trk::constants::apts::thickness; // fallback (no CCDB)
  auto& ccdbMgr = o2::ccdb::BasicCCDBManager::instance();
  ccdbMgr.setURL("http://alice-ccdb.cern.ch");
  if (auto* alpResp = ccdbMgr.get<o2::itsmft::AlpideSimResponse>("IT3/Calib/APTSResponse")) {
    depthMax = alpResp->getDepthMax();
  }
  const float yPlaneMLOT = o2::trk::SegmentationChip::SiliconThicknessMLOT / 2.f - depthMax;
  const float yPlaneVD = -o2::trk::SegmentationChip::SiliconThicknessVD; // VD reference plane in local flat y
  // Hits
  TFile* hitFile = TFile::Open(hitfile.data());
  TTree* hitTree = (TTree*)hitFile->Get("o2sim");
  int nevH = hitTree->GetEntries(); // hits are stored as one event per entry
  std::vector<std::vector<o2::trk::Hit>*> hitArray(nevH, nullptr);

  std::vector<std::unordered_map<uint64_t, int>> mc2hitVec(nevH);

  // Digits — per-layer branches
  TFile* digFile = TFile::Open(digifile.data());
  TTree* digTree = (TTree*)digFile->Get("o2sim");

  int nDigitLayers = 0;
  std::vector<std::vector<o2::itsmft::Digit>*> digArr(nTotalLayers, nullptr);
  std::vector<std::vector<o2::itsmft::ROFRecord>*> rofRecordsArr(nTotalLayers, nullptr);
  std::vector<o2::dataformats::IOMCTruthContainerView*> plabelsArr(nTotalLayers, nullptr);

  for (int iLayer = 0; iLayer < nTotalLayers; ++iLayer) {
    if (!digTree->GetBranch(Form("TRKDigit_%i", iLayer))) {
      break;
    }
    digTree->SetBranchAddress(Form("TRKDigit_%i", iLayer), &digArr[iLayer]);
    digTree->SetBranchAddress(Form("TRKDigitROF_%i", iLayer), &rofRecordsArr[iLayer]);
    digTree->SetBranchAddress(Form("TRKDigitMCTruth_%i", iLayer), &plabelsArr[iLayer]);
    ++nDigitLayers;
  }

  digTree->GetEntry(0);

  // Load all MC hit events upfront and build the hit lookup map.
  for (int im = 0; im < nevH; ++im) {
    hitTree->SetBranchAddress("TRKHit", &hitArray[im]);
    hitTree->GetEntry(im);
    auto& mc2hit = mc2hitVec[im];
    for (int ih = hitArray[im]->size(); ih--;) {
      const auto& hit = (*hitArray[im])[ih];
      uint64_t key = (uint64_t(hit.GetTrackID()) << 32) + hit.GetDetectorID();
      mc2hit.emplace(key, ih);
    }
  }

  // LOOP over layers, then ROFRecords within each layer
  for (int iLayer = 0; iLayer < nDigitLayers; ++iLayer) {
    auto& rofArr = *rofRecordsArr[iLayer];
    const int nROFRec = (int)rofArr.size();

    o2::dataformats::ConstMCTruthContainer<o2::MCCompLabel> labels;
    plabelsArr[iLayer]->copyandflatten(labels);

    // LOOP on : ROFRecord array
    for (unsigned int iROF = 0; iROF < rofArr.size(); ++iROF) {

      const unsigned int rofIndex = rofArr[iROF].getFirstEntry();
      const unsigned int rofNEntries = rofArr[iROF].getNEntries();

      // LOOP on : digits array
      for (unsigned int iDigit = rofIndex; iDigit < rofIndex + rofNEntries; iDigit++) {
        if (iDigit % 1000 == 0)
          std::cout << "Layer " << iLayer << ": reading digit " << iDigit << " / " << digArr[iLayer]->size() << std::endl;

        Int_t ix = (*digArr[iLayer])[iDigit].getRow(), iz = (*digArr[iLayer])[iDigit].getColumn();
        Int_t iDetID = (*digArr[iLayer])[iDigit].getChipIndex();
        Int_t layer = gman->getLayer(iDetID);
        Int_t disk = gman->getDisk(iDetID);
        Int_t subDetID = gman->getSubDetID(iDetID);
        Int_t petalCase = gman->getPetalCase(iDetID);
        Int_t stave = gman->getStave(iDetID);
        Int_t halfstave = gman->getHalfStave(iDetID);

        Float_t x = 0.f, y = 0.f, z = 0.f;
        Float_t x_flat = 0.f, z_flat = 0.f;

        if (disk != -1) {
          continue; // skip disks for the moment
        }

        if (subDetID != 0) {
          seg.detectorToLocal(ix, iz, x, z, subDetID, layer, disk);
        } else if (subDetID == 0) {
          seg.detectorToLocal(ix, iz, x_flat, z_flat, subDetID, layer, disk);
          o2::math_utils::Vector2D<float> xyCurved = seg.flatToCurved(layer, x_flat, 0.);
          x = xyCurved.X();
          y = xyCurved.Y();
          z = z_flat;
        }

        o2::math_utils::Point3D<float> locD(x, y, z);     // local Digit curved
        o2::math_utils::Point3D<float> locDF(-1, -1, -1); // local Digit flat

        Int_t chipID = (*digArr[iLayer])[iDigit].getChipIndex();
        auto lab = (labels.getLabels(iDigit))[0];

        int trID = lab.getTrackID();

        if (!lab.isValid()) { // not a noise
          continue;
        }

        const auto gloD = gman->getMatrixL2G(chipID)(locD); // convert to global

        std::unordered_map<uint64_t, int>* mc2hit = &mc2hitVec[lab.getEventID()];

        // get MC info
        uint64_t key = (uint64_t(trID) << 32) + chipID;
        auto hitEntry = mc2hit->find(key);

        if (hitEntry == mc2hit->end()) {
          LOG(error) << "Failed to find MC hit entry for Tr" << trID << " chipID" << chipID;
          continue;
        }

        ////// HITS
        Hit& hit = (*hitArray[lab.getEventID()])[hitEntry->second];

        auto xyzLocE = gman->getMatrixL2G(chipID) ^ (hit.GetPos()); // inverse conversion from global to local
        auto xyzLocS = gman->getMatrixL2G(chipID) ^ (hit.GetPosStart());

        // Hit local reference: Both VD and MLOT use response-plane interpolation (in flat local frame).
        // For VD, transform curved → flat first, then interpolate.
        o2::math_utils::Vector3D<float> locH;  /// Hit reference (at response plane)
        o2::math_utils::Vector3D<float> locHS; /// Hit, start pos
        locHS.SetCoordinates(xyzLocS.X(), xyzLocS.Y(), xyzLocS.Z());
        o2::math_utils::Vector3D<float> locHE; /// Hit, end pos
        locHE.SetCoordinates(xyzLocE.X(), xyzLocE.Y(), xyzLocE.Z());
        o2::math_utils::Vector3D<float> locHF;

        if (subDetID == 0) {
          // VD: Interpolate to VD reference plane in flat frame; apply same r to X and Z
          auto flatSta = seg.curvedToFlat(layer, locHS.X(), locHS.Y());
          auto flatEnd = seg.curvedToFlat(layer, locHE.X(), locHE.Y());
          float x0 = flatSta.X(), y0 = flatSta.Y(), z0 = locHS.Z();
          float dltx = flatEnd.X() - x0, dlty = flatEnd.Y() - y0, dltz = locHE.Z() - z0;
          float r = (std::abs(dlty) > 1e-9f) ? (yPlaneVD - y0) / dlty : 0.5f;
          locH.SetCoordinates(x0 + r * dltx, yPlaneVD, z0 + r * dltz);
        } else {
          // MLOT: Interpolate to response plane
          float x0 = locHS.X(), y0 = locHS.Y(), z0 = locHS.Z();
          float dltx = locHE.X() - x0, dlty = locHE.Y() - y0, dltz = locHE.Z() - z0;
          float r = (std::abs(dlty) > 1e-9f) ? (yPlaneMLOT - y0) / dlty : 0.5f;
          locH.SetCoordinates(x0 + r * dltx, yPlaneMLOT, z0 + r * dltz);
        }

        int row = 0, col = 0;
        float xlc = 0., zlc = 0.;

        if (subDetID == 0) {
          Float_t x_flat = 0.f, y_flat = 0.f;
          // locH is already in flat frame from interpolation above; convert digit to flat for comparison
          o2::math_utils::Vector2D<float> xyFlatD = seg.curvedToFlat(layer, locD.X(), locD.Y());
          locDF.SetCoordinates(xyFlatD.X(), xyFlatD.Y(), locD.Z());
          locHF.SetCoordinates(locH.X(), locH.Y(), locH.Z()); // locH already in flat frame
          seg.localToDetector(locHF.X(), locHF.Z(), row, col, subDetID, layer, disk);
        } else {
          seg.localToDetector(locH.X(), locH.Z(), row, col, subDetID, layer, disk);
        }

        seg.detectorToLocal(row, col, xlc, zlc, subDetID, layer, disk);

        if (subDetID == 0) {
          nt->Fill(chipID,                                                           /// detector ID
                   gloD.X(), gloD.Y(), gloD.Z(),                                     /// global position retrieved from the digit: digit (row, col) ->local position -> global potision
                   ix, iz,                                                           /// row and column of the digit
                   row, col,                                                         /// row and col retrieved from the hit: hit global position -> hit local position -> detector position (row, col)
                   locH.X(), locH.Z(),                                               /// x and z of the hit in the local reference frame: hit global position -> hit local position
                   xlc, zlc,                                                         /// x and z of the hit in the local frame: hit global position -> hit local position -> detector position (row, col) -> local position
                   locHF.X() - locDF.X(), locHF.Z() - locDF.Z());                    /// difference in x and z between the hit and the digit in the local frame
          nt2->Fill(chipID, gloD.Z(), locHS.X() - locHE.X(), locHS.Z() - locHE.Z()); /// differences between local hit start and hit end positions
        } else {
          nt->Fill(chipID,                                                           /// detector ID
                   gloD.X(), gloD.Y(), gloD.Z(),                                     /// global position retrieved from the digit: digit (row, col) ->local position -> global potision
                   ix, iz,                                                           /// row and column of the digit
                   row, col,                                                         /// row and col retrieved from the hit: hit global position -> hit local position -> detector position (row, col)
                   locH.X(), locH.Z(),                                               /// x and z of the hit in the local reference frame: hit global position -> hit local position
                   xlc, zlc,                                                         /// x and z of the hit in the local frame: hit global position -> hit local position -> detector position (row, col) -> local position
                   locH.X() - locD.X(), locH.Z() - locD.Z());                        /// difference in x and z between the hit and the digit in the local frame
          nt2->Fill(chipID, gloD.Z(), locHS.X() - locHE.X(), locHS.Z() - locHE.Z()); /// differences between local hit start and hit end positions
        }

      } // end loop on digits array

    } // end loop on ROFRecords

  } // end loop on layers

  // digit maps in the xy and yz planes
  auto canvXY = new TCanvas("canvXY", "", 1600, 2400);
  canvXY->Divide(2, 3);
  canvXY->cd(1);
  nt->Draw("y:x >>h_y_vs_x_VD(1000, -3, 3, 1000, -3, 3)", "id < 12 ", "colz");
  canvXY->cd(2);
  nt->Draw("y:z>>h_y_vs_z_VD(1000, -26, 26, 1000, -3, 3)", "id < 12 ", "colz");
  canvXY->cd(3);
  nt->Draw("y:x>>h_y_vs_x_ML(1000, -25, 25, 1000, -25, 25)", "id >= 12 && id < 5132 ", "colz");
  canvXY->cd(4);
  nt->Draw("y:z>>h_y_vs_z_ML(1000, -70, 70, 1000, -25, 25)", "id >= 12 && id < 5132 ", "colz");
  canvXY->cd(5);
  nt->Draw("y:x>>h_y_vs_x_OT(1000, -85, 85, 1000, -85, 85)", "id >= 5132 ", "colz");
  canvXY->cd(6);
  nt->Draw("y:z>>h_y_vs_z_OT(1000, -85, 85, 1000, -130, 130)", "id >= 5132 ", "colz");
  canvXY->SaveAs("trkdigits_y_vs_x_vs_z.pdf");

  // z distributions
  auto canvZ = new TCanvas("canvZ", "", 800, 2400);
  canvZ->Divide(1, 3);
  canvZ->cd(1);
  nt->Draw("z>>h_z_VD(500, -26, 26)", "id < 12 ");
  canvZ->cd(2);
  nt->Draw("z>>h_z_ML(500, -70, 70)", "id >= 12 && id < 5132 ");
  canvZ->cd(3);
  nt->Draw("z>>h_z_OT(500, -85, 85)", "id >= 5132 ");
  canvZ->SaveAs("trkdigits_z.pdf");

  // dz distributions (difference between local position of digits and hits in x and z)
  auto canvdZ = new TCanvas("canvdZ", "", 800, 2400);
  canvdZ->Divide(1, 3);
  canvdZ->cd(1);
  nt->Draw("dz>>h_dz_VD(500, -0.05, 0.05)", "id < 12 ");
  canvdZ->cd(2);
  nt->Draw("dz>>h_dz_ML(500, -0.05, 0.05)", "id >= 12 && id < 5132 ");
  canvdZ->cd(3);
  nt->Draw("dz>>h_dz_OT(500, -0.05, 0.05)", "id >= 5132 ");
  canvdZ->SaveAs("trkdigits_dz.pdf");
  canvdZ->SaveAs("trkdigits_dz.root");

  // distributions of differences between local positions of digits and hits in x and z
  auto canvdXdZ = new TCanvas("canvdXdZ", "", 1600, 2400);
  canvdXdZ->Divide(2, 3);
  canvdXdZ->cd(1);
  nt->Draw("dx:dz>>h_dx_vs_dz_VD(500, -0.005, 0.005, 500, -0.005, 0.005)", "id < 12", "colz");
  addTLines(o2::trk::SegmentationChip::PitchRowVD);
  auto h = (TH2F*)gPad->GetPrimitive("h_dx_vs_dz_VD");
  LOG(info) << "dx, dz";
  Info("VD", "RMS(dx)=%.1f mu", h->GetRMS(2) * 1e4);
  Info("VD", "RMS(dz)=%.1f mu", h->GetRMS(1) * 1e4);
  canvdXdZ->cd(2);
  nt->Draw("dx:dz>>h_dx_vs_dz_VD_z(500, -0.005, 0.005, 500, -0.005, 0.005)", "id < 12 && abs(z)<0.5", "colz");
  addTLines(o2::trk::SegmentationChip::PitchRowVD);
  h = (TH2F*)gPad->GetPrimitive("h_dx_vs_dz_VD_z");
  Info("VD |z|<1", "RMS(dx)=%.1f mu", h->GetRMS(2) * 1e4);
  Info("VD |z|<1", "RMS(dz)=%.1f mu", h->GetRMS(1) * 1e4);
  canvdXdZ->cd(3);
  nt->Draw("dx:dz>>h_dx_vs_dz_ML(600, -0.03, 0.03, 600, -0.03, 0.03)", "id >= 12 && id < 5132", "colz");
  addTLines(o2::trk::SegmentationChip::PitchRowMLOT);
  h = (TH2F*)gPad->GetPrimitive("h_dx_vs_dz_ML");
  Info("ML", "RMS(dx)=%.1f mu", h->GetRMS(2) * 1e4);
  Info("ML", "RMS(dz)=%.1f mu", h->GetRMS(1) * 1e4);
  canvdXdZ->cd(4);
  nt->Draw("dx:dz>>h_dx_vs_dz_ML_z(600, -0.03, 0.03, 600, -0.03, 0.03)", "id >= 12 && id < 5132 && abs(z)<2", "colz");
  addTLines(o2::trk::SegmentationChip::PitchRowMLOT);
  h = (TH2F*)gPad->GetPrimitive("h_dx_vs_dz_ML_z");
  Info("ML |z|<2", "RMS(dx)=%.1f mu", h->GetRMS(2) * 1e4);
  Info("ML |z|<2", "RMS(dz)=%.1f mu", h->GetRMS(1) * 1e4);
  canvdXdZ->SaveAs("trkdigits_dx_vs_dz.pdf");
  canvdXdZ->cd(5);
  nt->Draw("dx:dz>>h_dx_vs_dz_OT(600, -0.03, 0.03, 600, -0.03, 0.03)", "id >= 5132", "colz");
  addTLines(o2::trk::SegmentationChip::PitchRowMLOT);
  h = (TH2F*)gPad->GetPrimitive("h_dx_vs_dz_OT");
  Info("OT", "RMS(dx)=%.1f mu", h->GetRMS(2) * 1e4);
  Info("OT", "RMS(dz)=%.1f mu", h->GetRMS(1) * 1e4);
  canvdXdZ->cd(6);
  nt->Draw("dx:dz>>h_dx_vs_dz_OT_z(600, -0.03, 0.03, 600, -0.03, 0.03)", "id >= 5132 && abs(z)<2", "colz");
  h = (TH2F*)gPad->GetPrimitive("h_dx_vs_dz_OT_z");
  addTLines(o2::trk::SegmentationChip::PitchRowMLOT);
  Info("OT |z|<2", "RMS(dx)=%.1f mu", h->GetRMS(2) * 1e4);
  Info("OT |z|<2", "RMS(dz)=%.1f mu", h->GetRMS(1) * 1e4);
  canvdXdZ->SaveAs("trkdigits_dx_vs_dz.pdf");
  canvdXdZ->SaveAs("trkdigits_dx_vs_dz.root");

  // distribution of differences between hit start and hit end in local coordinates
  auto canvdXdZHit = new TCanvas("canvdXdZHit", "", 1600, 2400);
  canvdXdZHit->Divide(2, 3);
  canvdXdZHit->cd(1);
  nt2->Draw("dxH:dzH>>h_dxH_vs_dzH_VD(300, -0.03, 0.03, 300, -0.03, 0.03)", "id < 12", "colz");
  addTLines(o2::trk::SegmentationChip::PitchRowVD);
  LOG(info) << "dxH, dzH";
  h = (TH2F*)gPad->GetPrimitive("h_dxH_vs_dzH_VD");
  Info("VD", "RMS(dxH)=%.1f mu", h->GetRMS(2) * 1e4);
  Info("VD", "RMS(dzH)=%.1f mu", h->GetRMS(1) * 1e4);
  canvdXdZHit->cd(2);
  nt2->Draw("dxH:dzH>>h_dxH_vs_dzH_VD_z(300, -0.03, 0.03, 300, -0.03, 0.03)", "id < 12 && abs(z)<2", "colz");
  addTLines(o2::trk::SegmentationChip::PitchRowVD);
  h = (TH2F*)gPad->GetPrimitive("h_dxH_vs_dzH_VD_z");
  Info("VD |z|<2", "RMS(dxH)=%.1f mu", h->GetRMS(2) * 1e4);
  Info("VD |z|<2", "RMS(dzH)=%.1f mu", h->GetRMS(1) * 1e4);
  canvdXdZHit->cd(3);
  nt2->Draw("dxH:dzH>>h_dxH_vs_dzH_ML(300, -0.03, 0.03, 300, -0.03, 0.03)", "id >= 12 && id < 5132", "colz");
  addTLines(o2::trk::SegmentationChip::PitchRowMLOT);
  h = (TH2F*)gPad->GetPrimitive("h_dxH_vs_dzH_ML");
  Info("ML", "RMS(dxH)=%.1f mu", h->GetRMS(2) * 1e4);
  Info("ML", "RMS(dzH)=%.1f mu", h->GetRMS(1) * 1e4);
  canvdXdZHit->cd(4);
  nt2->Draw("dxH:dzH>>h_dxH_vs_dzH_ML_z(300, -0.03, 0.03, 300, -0.03, 0.03)", "id >= 12 && id < 5132 && abs(z)<2", "colz");
  addTLines(o2::trk::SegmentationChip::PitchRowMLOT);
  h = (TH2F*)gPad->GetPrimitive("h_dxH_vs_dzH_ML_z");
  Info("ML |z|<2", "RMS(dxH)=%.1f mu", h->GetRMS(2) * 1e4);
  Info("ML |z|<2", "RMS(dzH)=%.1f mu", h->GetRMS(1) * 1e4);
  canvdXdZHit->SaveAs("trkdigits_dxH_vs_dzH.pdf");
  canvdXdZHit->cd(5);
  nt2->Draw("dxH:dzH>>h_dxH_vs_dzH_OT(300, -0.03, 0.03, 300, -0.03, 0.03)", "id >= 5132", "colz");
  addTLines(o2::trk::SegmentationChip::PitchRowMLOT);
  h = (TH2F*)gPad->GetPrimitive("h_dxH_vs_dzH_OT");
  Info("OT", "RMS(dxH)=%.1f mu", h->GetRMS(2) * 1e4);
  Info("OT", "RMS(dzH)=%.1f mu", h->GetRMS(1) * 1e4);
  canvdXdZHit->cd(6);
  nt2->Draw("dxH:dzH>>h_dxH_vs_dzH_OT_z(300, -0.03, 0.03, 300, -0.03, 0.03)", "id >= 5132 && abs(z)<2", "colz");
  addTLines(o2::trk::SegmentationChip::PitchRowMLOT);
  h = (TH2F*)gPad->GetPrimitive("h_dxH_vs_dzH_OT_z");
  Info("OT |z|<2", "RMS(dxH)=%.1f mu", h->GetRMS(2) * 1e4);
  Info("OT |z|<2", "RMS(dzH)=%.1f mu", h->GetRMS(1) * 1e4);
  canvdXdZHit->SaveAs("trkdigits_dxH_vs_dzH.pdf");

  f->Write();
  f->Close();
}
