//
//  HitAnalysis.cpp
//  ALICEO2
//
//  Created by Markus Fasel on 28.07.15.
//
//

#include "HitAnalysis/HitAnalysis.h"

#include "FairLogger.h"               // for LOG

#include "TClonesArray.h"             // for TClonesArray
#include "TH1.h"                      // for TH1, TH1D, TH1F
#include "TMath.h"

#include "ITSSimulation/Chip.h"
#include "ITSSimulation/Point.h"
#include "ITSBase/Segmentation.h"
#include "ITSBase/GeometryTGeo.h"

using namespace AliceO2::ITS;

HitAnalysis::HitAnalysis() :
  FairTask(),
  fIsInitialized(kFALSE),
  fProcessChips(kTRUE),
  fChips(),
  fLocalX0(nullptr),
  fLocalX1(nullptr),
  fLocalY0(nullptr),
  fLocalY1(nullptr),
  fLocalZ0(nullptr),
  fLocalZ1(nullptr),
  fHitCounter(nullptr)
{ }

HitAnalysis::~HitAnalysis()
{
  // Delete chips
  for (std::map<int, Chip *>::iterator chipiter = fChips.begin(); chipiter != fChips.end(); ++chipiter) {
    delete chipiter->second;
  }
  // Delete geometry
  delete fGeometry;
  // Delete histograms
  delete fLineSegment;
  delete fLocalX0;
  delete fLocalX1;
  delete fLocalY0;
  delete fLocalY1;
  delete fLocalZ0;
  delete fLocalZ1;
}

InitStatus HitAnalysis::Init()
{
  // Get the FairRootManager
  FairRootManager *mgr = FairRootManager::Instance();
  if (!mgr) {
    LOG(ERROR) << "Could not instantiate FairRootManager. Exiting ..." << FairLogger::endl;
    return kERROR;
  }

  fPointsArray = dynamic_cast<TClonesArray *>(mgr->GetObject("ITSPoint"));
  if (!fPointsArray) {
    LOG(ERROR) << "ITS points not registered in the FairRootManager. Exiting ..." << FairLogger::endl;
    return kERROR;
  }

  // Create geometry, initialize chip array
  fGeometry = new GeometryTGeo(kTRUE, kTRUE);

  if (fProcessChips) {
    for (int chipid = 0; chipid < fGeometry->getNumberOfChips(); chipid++) {
      fChips[chipid] = new Chip(chipid, fGeometry);
    }
    LOG(DEBUG) << "Created " << fChips.size() << " chips." << FairLogger::endl;

    // Test whether indices match:
    LOG(DEBUG) << "Testing for integrity of chip indices" << FairLogger::endl;
    for (int i = 0; i < fGeometry->getNumberOfChips(); i++) {
      if (fChips[i]->GetChipIndex() != i) {
        LOG(ERROR) << "Chip index mismatch for entry " << i << ", value " << fChips[i]->GetChipIndex() <<
                   FairLogger::endl;
      }
    }
    LOG(DEBUG) << "Test for chip index integrity finished" << FairLogger::endl;
  }

  // Create histograms
  // Ranges to be adjusted
  Double_t maxLengthX(-1.), maxLengthY(-1.), maxLengthZ(-1.);
  const Segmentation *itsseg(nullptr);
  for (int ily = 0; ily < 7; ily++) {
    itsseg = fGeometry->getSegmentation(ily);
    if (itsseg->Dx() > maxLengthX) { maxLengthX = itsseg->Dx(); }
    if (itsseg->Dy() > maxLengthY) { maxLengthY = itsseg->Dy(); }
    if (itsseg->Dz() > maxLengthX) { maxLengthZ = itsseg->Dz(); }
  }
  fLineSegment = new TH1D("lineSegment", "Length of the line segment within the chip", 500, 0.0, 0.01);
  fLocalX0 = new TH1D("localX0", "X position in local (chip) coordinates at the start of a hit", 5000, -2 * maxLengthX,
                      2 * maxLengthX);
  fLocalX1 = new TH1D("localX1", "X position in local (chip) coordinates at the end of a hit", 500, -0.005, 0.005);
  fLocalY0 = new TH1D("localY0", "Y position in local (chip) coordinates at the start of a hit", 5000, -2 * maxLengthY,
                      2. * maxLengthY);
  fLocalY1 = new TH1D("localY1", "Y position in local (chip) coordinates at the end of a hit", 500, -0.005, 0.005);
  fLocalZ0 = new TH1D("localZ0", "Z position in local (chip) coordinates at the start of a hit", 5000, 2 * maxLengthZ,
                      -2 * maxLengthZ);
  fLocalZ1 = new TH1D("localZ1", "Z position in local (chip) coordinates at the end of a hit", 500, -0.005, 0.005);
  fHitCounter = new TH1F("hitcounter", "Simple hit counter", 1, 0.5, 1.5);

  fIsInitialized = kTRUE;
  return kSUCCESS;
}

void HitAnalysis::Exec(Option_t *option)
{
  if (!fIsInitialized) {
    return;
  }
  // Clear all chips
  //for (auto chipiter : fChips) {
  //  chipiter.second.Clear();
  //}

  // Add test: Count number of hits in the points array (cannot be larger then the entries in the tree)
  fHitCounter->Fill(1., fPointsArray->GetEntries());

  if (fProcessChips) {
    ProcessChips();
  } else {
    ProcessHits();
  }
}

void HitAnalysis::ProcessChips()
{
  // Add test: All chips must be empty
  Int_t nchipsNotEmpty(0);
  std::vector<int> nonEmptyChips;
  for (auto chipiter: fChips) {
    if (chipiter.second->GetNumberOfPoints() > 0) {
      nonEmptyChips.push_back(chipiter.second->GetChipIndex());
      nchipsNotEmpty++;
    }
  }
  if (nchipsNotEmpty) {
    LOG(ERROR) << "Number of non-empty chips larger 0: " << nchipsNotEmpty << FairLogger::endl;
    for (auto inditer : nonEmptyChips) {
      LOG(ERROR) << "Chip index " << inditer << FairLogger::endl;
    }
  }

  // Assign hits to chips
  for (TIter pointIter = TIter(fPointsArray).Begin(); pointIter != TIter::End(); ++pointIter) {
    Point *point = static_cast<Point *>(*pointIter);
    try {
      fChips[point->GetDetectorID()]->InsertPoint(point);
    } catch (Chip::IndexException &e) {
      LOG(ERROR) << e.what() << FairLogger::endl;
    }
  }

  // Add test: Total number of hits assigned to chips must be the same as the size of the points array
  Int_t nHitsAssigned(0);
  for (auto chipiter : fChips) {
    nHitsAssigned += chipiter.second->GetNumberOfPoints();
  }
  if (nHitsAssigned != fPointsArray->GetEntries()) {
    LOG(ERROR) << "Number of points mismatch: Read(" << fPointsArray->GetEntries() << "), Assigned(" << nHitsAssigned <<
               ")" << FairLogger::endl;
  }

  // Values for line segment calculation
  double x0, x1, y0, y1, z0, z1, tof, edep, steplength;

  // loop over chips, get the line segment
  for (auto chipiter: fChips) {
    Chip &mychip = *(chipiter.second);
    if (!mychip.GetNumberOfPoints()) {
      continue;
    }
    //LOG(DEBUG) << "Processing chip with index " << mychip.GetChipIndex() << FairLogger::endl;
    for (int ihit = 0; ihit < mychip.GetNumberOfPoints(); ihit++) {
      if (mychip[ihit]->IsEntering()) { continue; }
      mychip.LineSegmentLocal(ihit, x0, x1, y0, y1, z0, z1, tof, edep);
      steplength = TMath::Sqrt(x1 * x1 + y1 * y1 + z1 * z1);
      fLineSegment->Fill(steplength);
      fLocalX0->Fill(x0);
      fLocalX1->Fill(x1);
      fLocalY0->Fill(y0);
      fLocalY1->Fill(y1);
      fLocalZ0->Fill(z0);
      fLocalZ1->Fill(z1);
    }
  }
  // Clear all chips
  for (std::map<int, Chip *>::iterator chipiter = fChips.begin(); chipiter != fChips.end(); ++chipiter) {
    chipiter->second->Clear();
  }
}

void HitAnalysis::ProcessHits()
{
  for (TIter pointiter = TIter(fPointsArray).Begin(); pointiter != TIter::End(); ++pointiter) {
    Point *p = static_cast<Point *>(*pointiter);
    Double_t phitloc[3], pstartloc[3],
      phitglob[3] = {p->GetX(), p->GetY(), p->GetZ()},
      pstartglob[3] = {p->GetStartX(), p->GetStartY(), p->GetStartZ()};

    //fGeometry->GetMatrix(p->GetDetectorID())->MasterToLocal(phitglob, phitloc);
    //fGeometry->GetMatrix(p->GetDetectorID())->MasterToLocal(pstartglob, pstartloc);
    fGeometry->globalToLocal(p->GetDetectorID(), phitglob, phitloc);
    fGeometry->globalToLocal(p->GetDetectorID(), pstartglob, pstartloc);

    fLocalX0->Fill(pstartloc[0]);
    fLocalY0->Fill(pstartloc[1]);
    fLocalZ0->Fill(pstartloc[2]);
    fLocalX1->Fill(phitloc[0] - pstartloc[0]);
    fLocalY1->Fill(phitloc[1] - pstartloc[1]);
    fLocalZ1->Fill(phitloc[2] - pstartloc[2]);
    //fLocalX1->Fill(phitloc[0]);
    //fLocalY1->Fill(phitloc[1]);
    //fLocalZ1->Fill(phitloc[2]);
  }
}

void HitAnalysis::FinishTask()
{
  if (!fIsInitialized) {
    return;
  }

  TFile *outfile = TFile::Open("hitanalysis.root", "RECREATE");
  outfile->cd();
  fLineSegment->Write();
  fLocalX0->Write();
  fLocalX1->Write();
  fLocalY0->Write();
  fLocalY1->Write();
  fLocalZ0->Write();
  fLocalZ1->Write();
  fHitCounter->Write();
  outfile->Close();
  delete outfile;
}
