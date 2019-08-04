/// \file CheckDigits_mft.C
/// \brief Simple macro to check MFT digits

#if !defined(__CLING__) || defined(__ROOTCLING__)

#include <TCanvas.h>
#include <TFile.h>
#include <TH2F.h>
#include <TNtuple.h>
#include <TString.h>
#include <TTree.h>

#include "ITSMFTBase/Digit.h"
#include "ITSMFTBase/SegmentationAlpide.h"
#include "ITSMFTSimulation/Hit.h"
#include "MFTBase/GeometryTGeo.h"
#include "MathUtils/Utils.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include <vector>

#endif

using namespace o2::base;

void CheckDigits_mft(Int_t nEvents = 1, Int_t nMuons = 10, TString mcEngine = "TGeant3")
{
  using o2::itsmft::Digit;
  using o2::itsmft::Hit;
  using o2::itsmft::SegmentationAlpide;
  using namespace o2::mft;

  TH1F* hTrackID = new TH1F("hTrackID", "hTrackID", 1.1 * nMuons + 1, -0.5, (nMuons + 0.1 * nMuons) + 0.5);

  TFile* f = TFile::Open("CheckDigits.root", "recreate");
  TNtuple* nt = new TNtuple("ntd", "digit ntuple", "id:x:y:z:rowD:colD:rowH:colH:xlH:zlH:xlcH:zlcH:dx:dz");

  Char_t filename[100];

  // Geometry
  sprintf(filename, "AliceO2_%s.params_%iev_%imu.root", mcEngine.Data(), nEvents, nMuons);
  TFile* file = TFile::Open(filename);
  gFile->Get("FairGeoParSet");

  auto* gman = o2::mft::GeometryTGeo::Instance();
  gman->fillMatrixCache(o2::utils::bit2Mask(o2::TransformType::L2G));

  SegmentationAlpide seg;

  // Hits
  sprintf(filename, "AliceO2_%s.mc_%iev_%imu.root", mcEngine.Data(), nEvents, nMuons);
  TFile* file0 = TFile::Open(filename);
  std::cout << " Open hits file " << filename << std::endl;
  TTree* hitTree = (TTree*)gFile->Get("o2sim");
  std::vector<o2::itsmft::Hit>* hitArray = nullptr;
  hitTree->SetBranchAddress("MFTHit", &hitArray);

  // Digits
  sprintf(filename, "AliceO2_%s.digi_%iev_%imu.root", mcEngine.Data(), nEvents, nMuons);
  TFile* file1 = TFile::Open(filename);
  std::cout << " Open digits file " << filename << std::endl;
  TTree* digTree = (TTree*)gFile->Get("o2sim");
  std::vector<o2::itsmft::Digit>* digArr = nullptr;
  digTree->SetBranchAddress("MFTDigit", &digArr);
  o2::dataformats::MCTruthContainer<o2::MCCompLabel>* labels = nullptr;
  digTree->SetBranchAddress("MFTDigitMCTruth", &labels);

  Int_t nevD = digTree->GetEntries(); // digits in cont. readout may be grouped as few events per entry
  Int_t nevH = hitTree->GetEntries(); // hits are stored as one event per entry
  Int_t lastReadHitEv = -1;

  std::cout << "Found " << nevH << " events with hits " << std::endl;
  std::cout << "Found " << nevD << " events with digits " << std::endl;

  Int_t nNoise = 0;

  for (Int_t iev = 0; iev < nevD; iev++) {
    digTree->GetEvent(iev);
    Int_t nd = digArr->size();

    while (nd--) {
      const Digit* d = &(*digArr)[nd];
      Int_t ix = d->getRow(), iz = d->getColumn();
      Float_t x = 0.f, z = 0.f;
      seg.detectorToLocal(ix, iz, x, z);
      const Point3D<Float_t> locD(x, 0., z);

      const auto& labs = labels->getLabels(nd);

      Int_t chipID = d->getChipIndex();
      o2::MCCompLabel lab = labs[0];
      Int_t trID = lab.getTrackID();
      Int_t ievH = lab.getEventID();

      if (trID >= 0) { // not a noise

        const auto gloD = gman->getMatrixL2G(chipID)(locD); // convert to global
        Float_t dx = 0.f, dz = 0.f;

        if (lastReadHitEv != ievH) {
          hitTree->GetEvent(ievH);
          lastReadHitEv = ievH;
        }

        bool ok = false;
        for (auto& p : *hitArray) {
          if (p.GetDetectorID() != (Short_t)chipID)
            continue;
          if (p.GetTrackID() != (Int_t)lab)
            continue;

          auto locH = gman->getMatrixL2G(chipID) ^ (p.GetPos());         // inverse conversion from global to local
          auto locHsta = gman->getMatrixL2G(chipID) ^ (p.GetPosStart()); // ITS specific only
          locH.SetXYZ(0.5 * (locH.X() + locHsta.X()), 0.5 * (locH.Y() + locHsta.Y()), 0.5 * (locH.Z() + locHsta.Z()));
          Int_t row, col;
          Float_t xlc, zlc;
          seg.localToDetector(locH.X(), locH.Z(), row, col);
          seg.detectorToLocal(row, col, xlc, zlc);
          //
          nt->Fill(chipID, gloD.X(), gloD.Y(), gloD.Z(), ix, iz, row, col, locH.X(), locH.Z(), xlc, zlc,
                   locH.X() - locD.X(), locH.Z() - locD.Z());
          hTrackID->Fill((Float_t)p.GetTrackID());
          ok = true;
          break;

        } // hits

        if (!ok) {
          printf("did not find hit for digit %d in ev %d: MCEv:%d MCTrack %d\n", nd, iev, ievH, trID);
        }

      } else {
        nNoise++;
      } // not noise

    } // digits

  } // events

  printf("nt has %lld entries\n", nt->GetEntriesFast());

  TCanvas* c1 = new TCanvas("c1", "hTrackID", 50, 50, 600, 600);
  hTrackID->Scale(1. / (Float_t)nEvents);
  hTrackID->SetMinimum(0.);
  hTrackID->DrawCopy();

  new TCanvas;
  nt->Draw("y:x");
  new TCanvas;
  nt->Draw("dx:dz");
  f->Write();
  f->Close();

  printf("noise digits %d \n", nNoise);
}
