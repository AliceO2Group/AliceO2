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

/// \file CheckDigitsITS3.C
/// \brief Simple macro to check ITS3 digits

#if !defined(__CLING__) || defined(__ROOTCLING__)
#define ENABLE_UPGRADES
#include <TCanvas.h>
#include <TFile.h>
#include <TH2F.h>
#include <TNtuple.h>
#include <TString.h>
#include <TTree.h>
#include <TROOT.h>

#include <vector>
#include "ITSBase/GeometryTGeo.h"
#include "DataFormatsITSMFT/Digit.h"
#include "ITS3Base/SegmentationSuperAlpide.h"
#include "ITSMFTBase/SegmentationAlpide.h"
#include "ITSMFTSimulation/Hit.h"
#include "MathUtils/Utils.h"
#include "SimulationDataFormat/ConstMCTruthContainer.h"
#include "SimulationDataFormat/IOMCTruthContainerView.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "DetectorsBase/GeometryManager.h"

#include "DataFormatsITSMFT/ROFRecord.h"

#endif

void CheckDigitsITS3(std::string digifile = "it3digits.root", std::string hitfile = "o2sim_HitsIT3.root", std::string inputGeom = "o2sim_geometry.root", std::string paramfile = "o2sim_par.root", bool batch = true)
{
  gROOT->SetBatch(batch);

  using namespace o2::base;

  using o2::itsmft::Digit;
  using o2::itsmft::Hit;

  using o2::its3::SegmentationSuperAlpide;
  using o2::itsmft::SegmentationAlpide;

  TFile* f = TFile::Open("CheckDigits.root", "recreate");
  TNtuple* nt = new TNtuple("ntd", "digit ntuple", "id:x:y:z:rowD:colD:rowH:colH:xlH:zlH:xlcH:zlcH:dx:dz");

  // Geometry
  o2::base::GeometryManager::loadGeometry(inputGeom);
  auto* gman = o2::its::GeometryTGeo::Instance();
  gman->fillMatrixCache(o2::math_utils::bit2Mask(o2::math_utils::TransformType::L2G));

  // we assume that we have 2 chips per layer
  const int nChipsPerLayer = 2;

  std::vector<SegmentationSuperAlpide> segs{};
  for (int iLayer{0}; iLayer < gman->getNumberOfLayers() - 4; ++iLayer) {
    for (int iChip{0}; iChip < gman->getNumberOfChipsPerLayer(iLayer); ++iChip) {
      segs.push_back(SegmentationSuperAlpide(iLayer));
    }
  }

  // Hits
  TFile* hitFile = TFile::Open(hitfile.data());
  TTree* hitTree = (TTree*)hitFile->Get("o2sim");
  int nevH = hitTree->GetEntries(); // hits are stored as one event per entry
  std::vector<std::vector<o2::itsmft::Hit>*> hitArray(nevH, nullptr);

  std::vector<std::unordered_map<uint64_t, int>> mc2hitVec(nevH);

  // Digits
  TFile* digFile = TFile::Open(digifile.data());
  TTree* digTree = (TTree*)digFile->Get("o2sim");

  std::vector<o2::itsmft::Digit>* digArr = nullptr;
  digTree->SetBranchAddress("IT3Digit", &digArr);

  o2::dataformats::IOMCTruthContainerView* plabels = nullptr;
  digTree->SetBranchAddress("IT3DigitMCTruth", &plabels);

  int nevD = digTree->GetEntries(); // digits in cont. readout may be grouped as few events per entry

  int lastReadHitEv = -1;

  int nDigitRead = 0, nDigitFilled = 0;

  // Get Read Out Frame arrays
  std::vector<o2::itsmft::ROFRecord>* ROFRecordArrray = nullptr;
  digTree->SetBranchAddress("IT3DigitROF", &ROFRecordArrray);
  std::vector<o2::itsmft::ROFRecord>& ROFRecordArrrayRef = *ROFRecordArrray;

  std::vector<o2::itsmft::MC2ROFRecord>* MC2ROFRecordArrray = nullptr;
  digTree->SetBranchAddress("IT3DigitMC2ROF", &MC2ROFRecordArrray);
  std::vector<o2::itsmft::MC2ROFRecord>& MC2ROFRecordArrrayRef = *MC2ROFRecordArrray;

  digTree->GetEntry(0);

  int nROFRec = (int)ROFRecordArrrayRef.size();
  std::vector<int> mcEvMin(nROFRec, hitTree->GetEntries());
  std::vector<int> mcEvMax(nROFRec, -1);
  o2::dataformats::ConstMCTruthContainer<o2::MCCompLabel> labels;
  plabels->copyandflatten(labels);
  delete plabels;

  // >> build min and max MC events used by each ROF
  for (int imc = MC2ROFRecordArrrayRef.size(); imc--;) {
    const auto& mc2rof = MC2ROFRecordArrrayRef[imc];
    printf("MCRecord: ");
    mc2rof.print();

    if (mc2rof.rofRecordID < 0) {
      continue; // this MC event did not contribute to any ROF
    }

    for (int irfd = mc2rof.maxROF - mc2rof.minROF + 1; irfd--;) {

      int irof = mc2rof.rofRecordID + irfd;

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
  } // << build min and max MC events used by each ROF

  unsigned int rofIndex = 0;
  unsigned int rofNEntries = 0;

  // LOOP on : ROFRecord array
  for (unsigned int iROF = 0; iROF < ROFRecordArrrayRef.size(); iROF++) {

    rofIndex = ROFRecordArrrayRef[iROF].getFirstEntry();
    rofNEntries = ROFRecordArrrayRef[iROF].getNEntries();

    // >> read and map MC events contributing to this ROF
    for (int im = mcEvMin[iROF]; im <= mcEvMax[iROF]; im++) {

      if (!hitArray[im]) {

        hitTree->SetBranchAddress("IT3Hit", &hitArray[im]);
        hitTree->GetEntry(im);

        auto& mc2hit = mc2hitVec[im];

        for (int ih = hitArray[im]->size(); ih--;) {

          const auto& hit = (*hitArray[im])[ih];
          uint64_t key = (uint64_t(hit.GetTrackID()) << 32) + hit.GetDetectorID();
          mc2hit.emplace(key, ih);
        }
      }
    }

    // LOOP on : digits array
    for (unsigned int iDigit = rofIndex; iDigit < rofIndex + rofNEntries; iDigit++) {

      int ix = (*digArr)[iDigit].getRow(), iz = (*digArr)[iDigit].getColumn();
      float x{0.f}, y{0.f}, z{0.f};

      int chipID = (*digArr)[iDigit].getChipIndex();

      if (chipID < segs.size()) {
        float xFlat{0.f};
        segs[chipID].detectorToLocal(ix, iz, xFlat, z);
        segs[chipID].flatToCurved(xFlat, 0., x, y);
      } else {
        SegmentationAlpide::detectorToLocal(ix, iz, x, z);
      }

      const o2::math_utils::Point3D<float> locD(x, y, z);
      auto lab = (labels.getLabels(iDigit))[0];
      int trID = lab.getTrackID();

      if (lab.isValid()) { // not a noise

        nDigitRead++;

        auto gloD = gman->getMatrixL2G(chipID)(locD); // convert to global
        float dx = 0., dz = 0.;

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

        auto locH = gman->getMatrixL2G(chipID) ^ (hit.GetPos()); // inverse conversion from global to local
        auto locHsta = gman->getMatrixL2G(chipID) ^ (hit.GetPosStart());

        locH.SetXYZ(0.5 * (locH.X() + locHsta.X()), 0.5 * (locH.Y() + locHsta.Y()), 0.5 * (locH.Z() + locHsta.Z()));

        int row, col;
        float xlc = 0., zlc = 0.;

        if (chipID < segs.size()) {
          segs[chipID].localToDetector(locH.X(), locH.Z(), row, col);
          segs[chipID].detectorToLocal(row, col, xlc, zlc);
        } else {
          SegmentationAlpide::localToDetector(locH.X(), locH.Z(), row, col);
          SegmentationAlpide::detectorToLocal(row, col, xlc, zlc);
        }

        nt->Fill(chipID, gloD.X(), gloD.Y(), gloD.Z(), ix, iz, row, col, locH.X(), locH.Z(), xlc, zlc, locH.X() - locD.X(), locH.Z() - locD.Z());

        nDigitFilled++;
      } // not noise

    } // end loop on digits array

  } // end loop on ROFRecords array

  auto canvXY = new TCanvas("canvXY", "", 1600, 1600);
  canvXY->Divide(2, 2);
  canvXY->cd(1);
  nt->Draw("y:x>>h_y_vs_x_IB(1000, -10, 10, 1000, -10, 10)", "id < 6", "colz");
  canvXY->cd(2);
  nt->Draw("y:z>>h_y_vs_z_IB(1000, -15, 15, 1000, -10, 10)", "id < 6", "colz");
  canvXY->cd(3);
  nt->Draw("y:x>>h_y_vs_x_OB(1000, -50, 50, 1000, -50, 50)", "id >= 6", "colz");
  canvXY->cd(4);
  nt->Draw("y:z>>h_y_vs_z_OB(1000, -100, 100, 1000, -50, 50)", "id >= 6", "colz");
  canvXY->SaveAs("it3digits_y_vs_x_vs_z.pdf");

  auto canvdXdZ = new TCanvas("canvdXdZ", "", 1600, 800);
  canvdXdZ->Divide(2, 1);
  canvdXdZ->cd(1)->SetLogz();
  nt->Draw("dx:dz>>h_dx_vs_dz_IB(260, -0.026, 0.026, 260, -0.026, 0.026)", "id < 6", "colz");
  canvdXdZ->cd(2)->SetLogz();
  nt->Draw("dx:dz>>h_dx_vs_dz_OB(260, -0.026, 0.026, 260, -0.026, 0.026)", "id >= 6", "colz");
  canvdXdZ->SaveAs("it3digits_dx_vs_dz.pdf");

  f->Write();
  f->Close();
  printf("read %d filled %d\n", nDigitRead, nDigitFilled);
}
