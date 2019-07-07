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
#include "ITSMFTBase/Digit.h"
#include "ITSMFTBase/SegmentationAlpide.h"
#include "ITSMFTSimulation/Hit.h"
#include "MathUtils/Utils.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "DetectorsBase/GeometryManager.h"
#endif

using namespace o2::base;

void CheckDigits(std::string digifile = "itsdigits.root", std::string hitfile = "o2sim.root", std::string inputGeom = "O2geometry.root", std::string paramfile = "o2sim_par.root")
{
  using o2::itsmft::Digit;
  using o2::itsmft::Hit;
  using o2::itsmft::SegmentationAlpide;
  using namespace o2::its;

  TFile* f = TFile::Open("CheckDigits.root", "recreate");
  TNtuple* nt = new TNtuple("ntd", "digit ntuple", "id:x:y:z:rowD:colD:rowH:colH:xlH:zlH:xlcH:zlcH:dx:dz");

  // Geometry
  o2::base::GeometryManager::loadGeometry(inputGeom, "FAIRGeom");
  auto* gman = o2::its::GeometryTGeo::Instance();
  gman->fillMatrixCache(o2::utils::bit2Mask(o2::TransformType::L2G));

  SegmentationAlpide seg;

  // Hits
  TFile* file0 = TFile::Open(hitfile.data());
  TTree* hitTree = (TTree*)gFile->Get("o2sim");
  std::vector<o2::itsmft::Hit>* hitArray = nullptr;
  hitTree->SetBranchAddress("ITSHit", &hitArray);

  // Digits
  TFile* file1 = TFile::Open(digifile.data());
  TTree* digTree = (TTree*)gFile->Get("o2sim");
  std::vector<o2::itsmft::Digit>* digArr = nullptr;
  o2::dataformats::MCTruthContainer<o2::MCCompLabel>* labels = nullptr;
  digTree->SetBranchAddress("ITSDigit", &digArr);
  digTree->SetBranchAddress("ITSDigitMCTruth", &labels);

  int nevD = digTree->GetEntries(); // digits in cont. readout may be grouped as few events per entry
  int nevH = hitTree->GetEntries(); // hits are stored as one event per entry
  int lastReadHitEv = -1;

  int ndr = 0, ndf = 0;

  for (int iev = 0; iev < nevD; iev++) {
    digTree->GetEvent(iev);

    int nd = -1;
    for (const auto& d : *digArr) {
      nd++;
      Int_t ix = d.getRow(), iz = d.getColumn();
      Float_t x = 0.f, z = 0.f;
      seg.detectorToLocal(ix, iz, x, z);
      const Point3D<float> locD(x, 0., z);

      Int_t chipID = d.getChipIndex();
      const auto& labs = labels->getLabels(nd);
      int trID = labs[0].getTrackID();
      int ievH = labs[0].getEventID();

      if (labs[0].isValid()) { // not a noise
        ndr++;
        const auto gloD = gman->getMatrixL2G(chipID)(locD); // convert to global
        float dx = 0., dz = 0.;

        if (lastReadHitEv != ievH) {
          hitTree->GetEvent(ievH);
          lastReadHitEv = ievH;
        }
        bool ok = false;
        for (auto& p : *hitArray) {
          if (p.GetDetectorID() != chipID)
            continue;
          if (p.GetTrackID() != trID)
            continue;
          auto locH = gman->getMatrixL2G(chipID) ^ (p.GetPos()); // inverse conversion from global to local
          auto locHsta = gman->getMatrixL2G(chipID) ^ (p.GetPosStart());
          locH.SetXYZ(0.5 * (locH.X() + locHsta.X()), 0.5 * (locH.Y() + locHsta.Y()), 0.5 * (locH.Z() + locHsta.Z()));
          int row, col;
          float xlc, zlc;
          seg.localToDetector(locH.X(), locH.Z(), row, col);
          seg.detectorToLocal(row, col, xlc, zlc);
          //
          nt->Fill(chipID, gloD.X(), gloD.Y(), gloD.Z(), ix, iz, row, col, locH.X(), locH.Z(), xlc, zlc,
                   locH.X() - locD.X(), locH.Z() - locD.Z());
          ok = true;
          ndf++;
          break;
        }
        if (!ok) {
          printf("did not find hit for digit %d in ev %d: MCEv:%d MCTrack %d\n", nd, iev, ievH, trID);
        }
      }
    }
  }
  new TCanvas;
  nt->Draw("y:x");
  new TCanvas;
  nt->Draw("dx:dz");
  f->Write();
  f->Close();
  printf("read %d filled %d\n", ndr, ndf);
}
