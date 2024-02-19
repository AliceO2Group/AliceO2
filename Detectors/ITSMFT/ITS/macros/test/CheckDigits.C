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
/// \brief Simple macro to check ITSU digits

#if !defined(__CLING__) || defined(__ROOTCLING__)
#include <TCanvas.h>
#include <TFile.h>
#include <TH2F.h>
#include <TNtuple.h>
#include <TString.h>
#include <TTree.h>

#include <vector>
#include "ITSBase/GeometryTGeo.h"
#include "DataFormatsITSMFT/Digit.h"
#include "ITSMFTBase/SegmentationAlpide.h"
#include "ITSMFTSimulation/Hit.h"
#include "MathUtils/Utils.h"
#include "SimulationDataFormat/ConstMCTruthContainer.h"
#include "SimulationDataFormat/IOMCTruthContainerView.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "DetectorsBase/GeometryManager.h"

#include "DataFormatsITSMFT/ROFRecord.h"

#endif

void CheckDigits(std::string digifile = "itsdigits.root", std::string hitfile = "o2sim_HitsITS.root", std::string inputGeom = "", std::string paramfile = "o2sim_par.root")
{

  using namespace o2::base;
  using namespace o2::its;

  using o2::itsmft::Digit;
  using o2::itsmft::Hit;

  using o2::itsmft::SegmentationAlpide;

  TFile* f = TFile::Open("CheckDigits.root", "recreate");

  TNtuple* nt = new TNtuple("ntd", "digit ntuple", "id:x:y:z:rowD:colD:rowH:colH:xlH:zlH:xlcH:zlcH:dx:dz");

  // Geometry
  o2::base::GeometryManager::loadGeometry(inputGeom);
  auto* gman = o2::its::GeometryTGeo::Instance();
  gman->fillMatrixCache(o2::math_utils::bit2Mask(o2::math_utils::TransformType::L2G));

  SegmentationAlpide seg;

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
  digTree->SetBranchAddress("ITSDigit", &digArr);

  o2::dataformats::IOMCTruthContainerView* plabels = nullptr;
  digTree->SetBranchAddress("ITSDigitMCTruth", &plabels);

  int nevD = digTree->GetEntries(); // digits in cont. readout may be grouped as few events per entry

  int lastReadHitEv = -1;

  int nDigitRead = 0, nDigitFilled = 0;

  // Get Read Out Frame arrays
  std::vector<o2::itsmft::ROFRecord>* ROFRecordArrray = nullptr;
  digTree->SetBranchAddress("ITSDigitROF", &ROFRecordArrray);
  std::vector<o2::itsmft::ROFRecord>& ROFRecordArrrayRef = *ROFRecordArrray;

  std::vector<o2::itsmft::MC2ROFRecord>* MC2ROFRecordArrray = nullptr;
  digTree->SetBranchAddress("ITSDigitMC2ROF", &MC2ROFRecordArrray);
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

        hitTree->SetBranchAddress("ITSHit", &hitArray[im]);
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

      Int_t ix = (*digArr)[iDigit].getRow(), iz = (*digArr)[iDigit].getColumn();
      Float_t x = 0.f, z = 0.f;

      seg.detectorToLocal(ix, iz, x, z);

      const o2::math_utils::Point3D<float> locD(x, 0., z);

      Int_t chipID = (*digArr)[iDigit].getChipIndex();
      auto lab = (labels.getLabels(iDigit))[0];

      int trID = lab.getTrackID();

      if (lab.isValid()) { // not a noise

        nDigitRead++;

        const auto gloD = gman->getMatrixL2G(chipID)(locD); // convert to global
        float dx = 0., dz = 0.;

        std::unordered_map<uint64_t, int>* mc2hit = &mc2hitVec[lab.getEventID()];

        // get MC info
        uint64_t key = (uint64_t(trID) << 32) + chipID;
        auto hitEntry = mc2hit->find(key);

        if (hitEntry == mc2hit->end()) {

          LOG(error) << "Failed to find MC hit entry for Tr" << trID << " chipID" << chipID;
          continue;
        }

        Hit& hit = (*hitArray[lab.getEventID()])[hitEntry->second];

        auto locH = gman->getMatrixL2G(chipID) ^ (hit.GetPos()); // inverse conversion from global to local
        auto locHsta = gman->getMatrixL2G(chipID) ^ (hit.GetPosStart());

        locH.SetXYZ(0.5 * (locH.X() + locHsta.X()), 0.5 * (locH.Y() + locHsta.Y()), 0.5 * (locH.Z() + locHsta.Z()));

        int row, col;
        float xlc = 0., zlc = 0.;

        seg.localToDetector(locH.X(), locH.Z(), row, col);
        seg.detectorToLocal(row, col, xlc, zlc);

        nt->Fill(chipID, gloD.X(), gloD.Y(), gloD.Z(), ix, iz, row, col, locH.X(), locH.Z(), xlc, zlc, locH.X() - locD.X(), locH.Z() - locD.Z());

        nDigitFilled++;
      }

    } // end loop on digits array

  } // end loop on ROFRecords array

  new TCanvas;
  nt->Draw("y:x");
  new TCanvas;
  nt->Draw("dx:dz", "abs(dx)<0.01 && abs(dz)<0.01");

  f->Write();
  f->Close();
  printf("read %d filled %d\n", nDigitRead, nDigitFilled);
}
