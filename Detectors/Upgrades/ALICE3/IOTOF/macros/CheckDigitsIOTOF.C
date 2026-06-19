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

/// \file CheckDigitsIOTOF.C
/// \brief Simple macro to check TF3 digits

#if !defined(__CLING__) || defined(__ROOTCLING__)
#include <TCanvas.h>
#include <TFile.h>
#include <TH2F.h>
#include <TNtuple.h>
#include <TString.h>
#include <TTree.h>
#include <TLine.h>
#include <TStyle.h>

#include "IOTOFSimulation/Segmentation.h"
#include "IOTOFBase/IOTOFBaseParameters.h"
#include "IOTOFBase/GeometryTGeo.h"
#include "DataFormatsIOTOF/Digit.h"
#include "ITSMFTSimulation/Hit.h"
#include "MathUtils/Utils.h"
#include "SimulationDataFormat/ConstMCTruthContainer.h"
#include "SimulationDataFormat/IOMCTruthContainerView.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "DetectorsBase/GeometryManager.h"
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

void CheckDigitsIOTOF(std::string digifile = "tf3digits.root", std::string hitfile = "o2sim_HitsTF3.root", std::string inputGeom = "o2sim_geometry.root")
{
  gStyle->SetPalette(55);

  using namespace o2::base;
  using namespace o2::iotof;

  using o2::iotof::Digit;
  using o2::itsmft::Hit;

  o2::conf::ConfigurableParam::updateFromString("IOTOFBase.segmentedInnerTOF=true;IOTOFBase.segmentedOuterTOF=true;IOTOFBase.enableForwardTOF=false;IOTOFBase.enableBackwardTOF=false");

  auto seg = o2::iotof::Segmentation::Instance();

  TFile* f = TFile::Open("CheckDigits.root", "recreate");

  TNtuple* nt = new TNtuple("ntd", "digit ntuple", "id:x:y:z:rowD:colD:rowH:colH:xlH:zlH:xlcH:zlcH:dx:dz");
  TNtuple* nt2 = new TNtuple("ntd2", "digit ntuple", "id:z:dxH:dzH"); /// maximum number of elements in a tuple = 15: doing a new tuple to store more variables

  auto& iotofPars = IOTOFBaseParam::Instance();

  // Geometry
  o2::base::GeometryManager::loadGeometry(inputGeom);
  auto* gman = o2::iotof::GeometryTGeo::Instance();
  gman->fillMatrixCache(o2::math_utils::bit2Mask(o2::math_utils::TransformType::L2G));

  // IOTOF response plane y = half sensor thickness
  float depthMax = iotofPars.sensorThickness; // fallback (no CCDB)
  const float yPlaneIOTOF = depthMax / 2.;

  // Hits
  TFile* hitFile = TFile::Open(hitfile.data());
  TTree* hitTree = (TTree*)hitFile->Get("o2sim");
  int nevH = hitTree->GetEntries(); // hits are stored as one event per entry
  std::vector<std::vector<o2::itsmft::Hit>*> hitArray(nevH, nullptr);

  std::vector<std::unordered_map<uint64_t, int>> mc2hitVec(nevH);

  // Digits
  TFile* digFile = TFile::Open(digifile.data());
  TTree* digTree = (TTree*)digFile->Get("o2sim");

  std::vector<o2::iotof::Digit>* digArr{nullptr};
  std::vector<o2::itsmft::ROFRecord>* rofRecordsArr{nullptr};
  o2::dataformats::IOMCTruthContainerView* plabelsArr{nullptr};

  digTree->SetBranchAddress("TF3Digit", &digArr);
  digTree->SetBranchAddress("TF3DigitROF", &rofRecordsArr);
  digTree->SetBranchAddress("TF3DigitMCTruth", &plabelsArr);

  digTree->GetEntry(0);

  // Load all MC hit events upfront and build the hit lookup map.
  for (int im = 0; im < nevH; ++im) {
    hitTree->SetBranchAddress("TF3Hit", &hitArray[im]);
    hitTree->GetEntry(im);
    auto& mc2hit = mc2hitVec[im];
    for (int ih = hitArray[im]->size(); ih--;) {
      const auto& hit = (*hitArray[im])[ih];
      uint64_t key = (uint64_t(hit.GetTrackID()) << 32) + hit.GetDetectorID();
      mc2hit.emplace(key, ih);
    }
  }

  auto& rofArr = *rofRecordsArr;
  const int nROFRec = (int)rofArr.size();

  o2::dataformats::ConstMCTruthContainer<o2::MCCompLabel> labels;
  plabelsArr->copyandflatten(labels);

  // LOOP on : ROFRecord array
  for (unsigned int iROF = 0; iROF < rofArr.size(); ++iROF) {

    const unsigned int rofIndex = rofArr[iROF].getFirstEntry();
    const unsigned int rofNEntries = rofArr[iROF].getNEntries();

    // LOOP on : digits array
    for (unsigned int iDigit = rofIndex; iDigit < rofIndex + rofNEntries; iDigit++) {
      if (iDigit % 1000 == 0) {
        std::cout << "Reading digit " << iDigit << " / " << digArr->size() << std::endl;
      }

      Int_t ix = (*digArr)[iDigit].getRow(), iz = (*digArr)[iDigit].getColumn();
      Int_t iDetID = (*digArr)[iDigit].getChipIndex();
      Int_t subDetID = gman->getIOTOFLayer(iDetID);

      Float_t x = 0.f, y = 0.f, z = 0.f;
      Float_t x_flat = 0.f, z_flat = 0.f;

      // Float_t t = (*digArr)[iDigit].getTime();

      if (subDetID >= 0) {
        seg->detectorToLocal(ix, iz, x, z, subDetID);
      }

      o2::math_utils::Point3D<float> locD(x, y, z);     // local Digit

      Int_t chipID = (*digArr)[iDigit].getChipIndex();

      auto lab = (labels.getLabels(iDigit))[0];

      if (!lab.isValid()) { // not a noise
        continue;
      }

      int trID = lab.getTrackID();

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

      // Hit local reference: use response plane interpolation
      o2::math_utils::Vector3D<float> locH;  /// Hit reference (at response plane)
      o2::math_utils::Vector3D<float> locHS; /// Hit, start pos
      locHS.SetCoordinates(xyzLocS.X(), xyzLocS.Y(), xyzLocS.Z());
      o2::math_utils::Vector3D<float> locHE; /// Hit, end pos
      locHE.SetCoordinates(xyzLocE.X(), xyzLocE.Y(), xyzLocE.Z());
      o2::math_utils::Vector3D<float> locHF;

      // IOTOF: Interpolate to response plane
      float x0 = locHS.X(), y0 = locHS.Y(), z0 = locHS.Z();
      float dltx = locHE.X() - x0, dlty = locHE.Y() - y0, dltz = locHE.Z() - z0;
      float r = (std::abs(dlty) > 1e-9f) ? (yPlaneIOTOF - y0) / dlty : 0.5f;
      locH.SetCoordinates(x0 + r * dltx, yPlaneIOTOF, z0 + r * dltz);

      int row = 0, col = 0;
      float xlc = 0., zlc = 0.;

      seg->localToDetector(locH.X(), locH.Z(), row, col, subDetID);
      seg->detectorToLocal(row, col, xlc, zlc, subDetID);
      nt->Fill(chipID,                                                           /// detector ID
               gloD.X(), gloD.Y(), gloD.Z(),                                     /// global position retrieved from the digit: digit (row, col) ->local position -> global potision
               ix, iz,                                                           /// row and column of the digit
               row, col,                                                         /// row and col retrieved from the hit: hit global position -> hit local position -> detector position (row, col)
               locH.X(), locH.Z(),                                               /// x and z of the hit in the local reference frame: hit global position -> hit local position
               xlc, zlc,                                                         /// x and z of the hit in the local frame: hit global position -> hit local position -> detector position (row, col) -> local position
               locHS.X() - locD.X(), locHS.Z() - locD.Z());                        /// difference in x and z between the hit and the digit in the local frame
      nt2->Fill(chipID, gloD.Z(), locHS.X() - locHE.X(), locHS.Z() - locHE.Z()); /// differences between local hit start and hit end positions

    } // end loop on digits array

  } // end loop on ROFRecords

  // digit maps in the xy and yz planes
  auto canvXY = new TCanvas("canvXY", "", 1600, 800);
  canvXY->Divide(2, 1);
  canvXY->cd(1);
  nt->Draw("y:x>>h_y_vs_x_IOTOF(1000, -100, 100, 1000, -100, 100)", "id >= 0 && id < 14352", "colz");
  canvXY->cd(2);
  nt->Draw("y:z>>h_y_vs_z_IOTOF(1000, -400, 400, 1000, -100, 100)", "id >= 0 && id < 14352", "colz");
  canvXY->SaveAs("tf3digits_y_vs_x_vs_z.pdf");

  // z distributions
  auto canvZ = new TCanvas("canvZ", "", 800, 800);
  canvZ->cd();
  nt->Draw("z>>h_z_IOTOF(500, -70, 70)", "id >= 0 && id < 14352 ");
  canvZ->SaveAs("tf3digits_z.pdf");

  // dz distributions (difference between local position of digits and hits in x and z)
  auto canvdZ = new TCanvas("canvdZ", "", 800, 800);
  canvdZ->cd();
  nt->Draw("dz>>h_dz_ML(500, -0.05, 0.05)", "id >= 0 && id < 14352 ");
  canvdZ->SaveAs("tf3digits_dz.pdf");
  canvdZ->SaveAs("tf3digits_dz.root");

  // distributions of differences between local positions of digits and hits in x and z
  auto canvdXdZ = new TCanvas("canvdXdZ", "", 1600, 800);
  canvdXdZ->Divide(2, 1);
  canvdXdZ->cd(1);
  nt->Draw("dx:dz>>h_dx_vs_dz_ITOF(600, -0.03, 0.03, 600, -0.03, 0.03)", "id >= 0 && id < 960", "colz");
  addTLines(0.01);
  auto h = (TH2F*)gPad->GetPrimitive("h_dx_vs_dz_ITOF");
  Info("ITOF", "RMS(dx)=%.1f mu", h->GetRMS(2) * 1e4);
  Info("ITOF", "RMS(dz)=%.1f mu", h->GetRMS(1) * 1e4);
  canvdXdZ->cd(2);
  nt->Draw("dx:dz>>h_dx_vs_dz_OTOF(600, -0.03, 0.03, 600, -0.03, 0.03)", "id >= 960 && id < 14352", "colz");
  addTLines(0.01);
  h = (TH2F*)gPad->GetPrimitive("h_dx_vs_dz_OTOF");
  Info("OTOF", "RMS(dx)=%.1f mu", h->GetRMS(2) * 1e4);
  Info("OTOF", "RMS(dz)=%.1f mu", h->GetRMS(1) * 1e4);
  canvdXdZ->SaveAs("tf3digits_dx_vs_dz.pdf");
  canvdXdZ->SaveAs("tf3digits_dx_vs_dz.root");

  // distribution of differences between hit start and hit end in local coordinates
  auto canvdXdZHit = new TCanvas("canvdXdZHit", "", 1600, 800);
  canvdXdZHit->Divide(2, 1);
  canvdXdZHit->cd(1);
  LOG(info) << "dxH, dzH";
  nt2->Draw("dxH:dzH>>h_dxH_vs_dzH_ITOF(300, -0.03, 0.03, 300, -0.03, 0.03)", "id >= 0 && id < 960", "colz");
  addTLines(0.01);
  h = (TH2F*)gPad->GetPrimitive("h_dxH_vs_dzH_ITOF");
  Info("ITOF", "RMS(dxH)=%.1f mu", h->GetRMS(2) * 1e4);
  Info("ITOF", "RMS(dzH)=%.1f mu", h->GetRMS(1) * 1e4);
  canvdXdZHit->cd(2);
  nt2->Draw("dxH:dzH>>h_dxH_vs_dzH_OTOF(300, -0.03, 0.03, 300, -0.03, 0.03)", "id >= 960 && id < 14352", "colz");
  addTLines(0.01);
  h = (TH2F*)gPad->GetPrimitive("h_dxH_vs_dzH_OTOF");
  Info("OTOF", "RMS(dxH)=%.1f mu", h->GetRMS(2) * 1e4);
  Info("OTOF", "RMS(dzH)=%.1f mu", h->GetRMS(1) * 1e4);
  canvdXdZHit->SaveAs("trkdigits_dxH_vs_dzH.pdf");

  f->Write();
  f->Close();
}
