// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

//
//  HitAnalysis.cpp
//  ALICEO2
//
//  Created by Markus Fasel on 28.07.15.
//
//

#include "HitAnalysis/HitAnalysis.h"

#include "FairLogger.h"               // for LOG

#include "TH1.h"                      // for TH1, TH1D, TH1F
#include "TMath.h"

#include "ITSMFTSimulation/Chip.h"
#include "ITSMFTSimulation/Hit.h"
#include "ITSMFTBase/SegmentationAlpide.h"
#include "ITSBase/GeometryTGeo.h"
#include "MathUtils/Cartesian3D.h"
#include "MathUtils/Utils.h"
#include <vector>

using Segmentation = o2::ITSMFT::SegmentationAlpide;
using o2::ITSMFT::Chip;
using o2::ITSMFT::Hit;

using namespace o2::ITS;
using namespace o2::Base;
using namespace o2::utils;

HitAnalysis::HitAnalysis() :
  FairTask(),
  mIsInitialized(kFALSE),
  mProcessChips(kTRUE),
  mChips(),
  mLocalX0(nullptr),
  mLocalX1(nullptr),
  mLocalY0(nullptr),
  mLocalY1(nullptr),
  mLocalZ0(nullptr),
  mLocalZ1(nullptr),
  mHitCounter(nullptr)
{ }

HitAnalysis::~HitAnalysis()
{
  // Delete chips
  for (std::map<int, Chip *>::iterator chipiter = mChips.begin(); chipiter != mChips.end(); ++chipiter) {
    delete chipiter->second;
  }
  // Delete geometry
  delete mGeometry;
  // Delete histograms
  delete mLineSegment;
  delete mLocalX0;
  delete mLocalX1;
  delete mLocalY0;
  delete mLocalY1;
  delete mLocalZ0;
  delete mLocalZ1;
}

InitStatus HitAnalysis::Init()
{
  // Get the FairRootManager
  FairRootManager *mgr = FairRootManager::Instance();
  if (!mgr) {
    LOG(ERROR) << "Could not instantiate FairRootManager. Exiting ..." << FairLogger::endl;
    return kERROR;
  }

  mHits = mgr->InitObjectAs<const std::vector<o2::ITSMFT::Hit>*>("ITSHit");
  if (!mHits) {
    LOG(ERROR) << "ITS points not registered in the FairRootManager. Exiting ..." << FairLogger::endl;
    return kERROR;
  }

  // Create geometry, initialize chip array
  GeometryTGeo* geom = GeometryTGeo::Instance();
  if ( !geom->isBuilt() ) geom->Build(true);
  geom->fillMatrixCache( o2::utils::bit2Mask(o2::TransformType::L2G) ); // make sure T2L matrices are loaded

  mGeometry = geom;

  if (mProcessChips) {
    for (int chipid = 0; chipid < mGeometry->getNumberOfChips(); chipid++) {
      mChips[chipid] = new Chip(nullptr, chipid, &mGeometry->getMatrixSensor(chipid));
    }
    LOG(DEBUG) << "Created " << mChips.size() << " chips." << FairLogger::endl;

    // Test whether indices match:
    LOG(DEBUG) << "Testing for integrity of chip indices" << FairLogger::endl;
    for (int i = 0; i < mGeometry->getNumberOfChips(); i++) {
      if (mChips[i]->GetChipIndex() != i) {
        LOG(ERROR) << "Chip index mismatch for entry " << i << ", value " << mChips[i]->GetChipIndex() <<
                   FairLogger::endl;
      }
    }
    LOG(DEBUG) << "Test for chip index integrity finished" << FairLogger::endl;
  }

  // Create histograms
  // Ranges to be adjusted
  Double_t maxLengthX(Segmentation::SensorSizeRows),
    maxLengthY(Segmentation::SensorThickness),
    maxLengthZ(Segmentation::SensorSizeCols);

  mLineSegment = new TH1D("lineSegment", "Length of the line segment within the chip", 500, 0.0, 0.01);
  mLocalX0 = new TH1D("localX0", "X position in local (chip) coordinates at the start of a hit", 5000, -2 * maxLengthX,
                      2 * maxLengthX);
  mLocalX1 = new TH1D("localX1", "X position in local (chip) coordinates at the end of a hit", 500, -0.005, 0.005);
  mLocalY0 = new TH1D("localY0", "Y position in local (chip) coordinates at the start of a hit", 5000, -2 * maxLengthY,
                      2. * maxLengthY);
  mLocalY1 = new TH1D("localY1", "Y position in local (chip) coordinates at the end of a hit", 500, -0.005, 0.005);
  mLocalZ0 = new TH1D("localZ0", "Z position in local (chip) coordinates at the start of a hit", 5000, 2 * maxLengthZ,
                      -2 * maxLengthZ);
  mLocalZ1 = new TH1D("localZ1", "Z position in local (chip) coordinates at the end of a hit", 500, -0.005, 0.005);
  mHitCounter = new TH1F("hitcounter", "Simple hit counter", 1, 0.5, 1.5);

  mIsInitialized = kTRUE;
  return kSUCCESS;
}

void HitAnalysis::Exec(Option_t *option)
{
  if (!mIsInitialized) {
    return;
  }
  // Clear all chips
  //for (auto chipiter : fChips) {
  //  chipiter.second.Clear();
  //}

  // Add test: Count number of hits in the points array (cannot be larger then the entries in the tree)
  mHitCounter->Fill(1., mHits->size());

  if (mProcessChips) {
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
  for (auto chipiter: mChips) {
    if (chipiter.second->GetNumberOfHits() > 0) {
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
  for(auto& hit : *mHits) {
    try {
      mChips[hit.GetDetectorID()]->InsertHit(&hit);
    } catch (Chip::IndexException &e) {
      LOG(ERROR) << e.what() << FairLogger::endl;
    }
  }

  // Add test: Total number of hits assigned to chips must be the same as the size of the points array
  Int_t nHitsAssigned(0);
  for (auto chipiter : mChips) {
    nHitsAssigned += chipiter.second->GetNumberOfHits();
  }
  if (nHitsAssigned != mHits->size()) {
    LOG(ERROR) << "Number of points mismatch: Read(" << mHits->size() << "), Assigned(" << nHitsAssigned <<
               ")" << FairLogger::endl;
  }

  // Values for line segment calculation
  double x0, x1, y0, y1, z0, z1, tof, edep, steplength;

  // loop over chips, get the line segment
  for (auto chipiter: mChips) {
    Chip &mychip = *(chipiter.second);
    if (!mychip.GetNumberOfHits()) {
      continue;
    }
    //LOG(DEBUG) << "Processing chip with index " << mychip.GetChipIndex() << FairLogger::endl;
    for (int ihit = 0; ihit < mychip.GetNumberOfHits(); ihit++) {
      if (mychip.GetHitAt(ihit)->IsEntering()) { continue; }
      mychip.LineSegmentLocal(ihit, x0, x1, y0, y1, z0, z1, tof, edep);
      steplength = TMath::Sqrt(x1 * x1 + y1 * y1 + z1 * z1);
      mLineSegment->Fill(steplength);
      mLocalX0->Fill(x0);
      mLocalX1->Fill(x1);
      mLocalY0->Fill(y0);
      mLocalY1->Fill(y1);
      mLocalZ0->Fill(z0);
      mLocalZ1->Fill(z1);
    }
  }
  // Clear all chips
  for (std::map<int, Chip *>::iterator chipiter = mChips.begin(); chipiter != mChips.end(); ++chipiter) {
    chipiter->second->Clear();
  }
}

void HitAnalysis::ProcessHits()
{
  for (auto& hit : *mHits) {
    auto loc = mGeometry->getMatrixL2G(hit.GetDetectorID())^(hit.GetPos()); // global->local end position
    auto locS = mGeometry->getMatrixL2G(hit.GetDetectorID())^(hit.GetPosStart()); // global->local start position
    auto glo = mGeometry->getMatrixL2G(hit.GetDetectorID())(loc);
    auto gloS = mGeometry->getMatrixL2G(hit.GetDetectorID())(locS);
    mLocalX0->Fill(locS.X());
    mLocalY0->Fill(locS.Y());
    mLocalZ0->Fill(locS.Z());
    loc.SetXYZ(loc.X()-locS.X(),loc.Y()-locS.Y(),loc.Z()-locS.Z());
    mLocalX1->Fill(loc.X());
    mLocalY1->Fill(loc.Y());
    mLocalZ1->Fill(loc.Z());
  }
}

void HitAnalysis::FinishTask()
{
  if (!mIsInitialized) {
    return;
  }

  TFile *outfile = TFile::Open("hitanalysis.root", "RECREATE");
  outfile->cd();
  mLineSegment->Write();
  mLocalX0->Write();
  mLocalX1->Write();
  mLocalY0->Write();
  mLocalY1->Write();
  mLocalZ0->Write();
  mLocalZ1->Write();
  mHitCounter->Write();
  outfile->Close();
  delete outfile;
}
