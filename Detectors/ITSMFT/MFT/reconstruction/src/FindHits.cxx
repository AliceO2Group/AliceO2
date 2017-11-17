// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file FindHits.h
/// \brief Simple hits finding from the points
/// \author bogdan.vulpescu@cern.ch 
/// \date 10/10/2016

#include "ITSMFTSimulation/Point.h"
#include "ITSMFTBase/SegmentationAlpide.h"

#include "MFTBase/Constants.h"
#include "MFTBase/Geometry.h"
#include "MFTSimulation/EventHeader.h"
#include "MFTReconstruction/Hit.h"
#include "MFTReconstruction/FindHits.h"

#include "TClonesArray.h"
#include "TMath.h"

#include <FairMQLogger.h>
#include "FairMCEventHeader.h"

using namespace o2::MFT;
using namespace o2::ITSMFT;

ClassImp(o2::MFT::FindHits)

//_____________________________________________________________________________
FindHits::FindHits():
mPoints(nullptr),
mHits(nullptr),
mNHits(0),
mTNofEvents(0),
mTNofHits(0),
mMCEventHeader(nullptr),
mEventHeader(nullptr)
{

}

//_____________________________________________________________________________
FindHits::~FindHits()
{

  reset();
  if (mHits) {
    mHits->Delete();
    delete mHits;
  }

}

//_____________________________________________________________________________
InitStatus FindHits::Init()
{

  LOG(INFO) << "FindHits::Init >>>>" << "";

  // Get RootManager
  FairRootManager* ioman = FairRootManager::Instance();
  if (!ioman) {
    //LOG(FATAL) << "RootManager not instantiated!" << "";
    return kFATAL;
  }

  // Get input array
  mPoints = static_cast<TClonesArray*>(ioman->GetObject("MFTPoints"));
  if (!mPoints) {
    //LOG(FATAL) << "No Point array!" << "";
    return kFATAL;
  }

  // Create and register output array
  mHits = new TClonesArray("o2::MFT::Hit");
  ioman->Register("MFTHits", "MFT", mHits, kTRUE);

  mEventHeader = new EventHeader();
  mEventHeader->SetName("EventHeader.");
  ioman->Register("EventHeader.","EvtHeader", mEventHeader, kFALSE);

  return kSUCCESS;

}

//_____________________________________________________________________________
InitStatus FindHits::ReInit()
{

  LOG(DEBUG) << "Re-Initilization of FindHits" << "";

  return kSUCCESS;

}

//_____________________________________________________________________________
void FindHits::initMQ(TList* tempList) 
{

  LOG(INFO) << "FindHits::InitMQ >>>>>" << "";

  mEventHeader = new EventHeader();
  mEventHeader->SetName("EventHeader.");
  mHits = new TClonesArray("o2::MFT::Hit",10000);

  return;

}

//_____________________________________________________________________________
void FindHits::Exec(Option_t* /*opt*/) 
{

  //Info("Exec","Exec called",0,0);
  LOG(INFO) << "FindHits::Exec >>>>>" << "";

  reset();

  o2::ITSMFT::Point *point;
  TVector3 pos, dpos;
  Int_t detID, trackID;
  Double_t dx = SegmentationAlpide::PitchRow/TMath::Sqrt(12);
  Double_t dy = SegmentationAlpide::PitchCol/TMath::Sqrt(12);
  Double_t dz = 0.;

  // Loop over fPoints
  Int_t nPoints = mPoints->GetEntriesFast();
  for (Int_t iPoint = 0; iPoint < nPoints; iPoint++) {
    point = static_cast<o2::ITSMFT::Point*>(mPoints->At(iPoint));
    if (!point) continue;
    detID = point->GetDetectorID();
    trackID = point->GetTrackID();
    // copy the coordinates from point to hit
    pos.SetXYZ(point->GetStartX(),point->GetStartY(),point->GetStartZ());
    dpos.SetXYZ(dx,dy,dz);
    //new ((*fHits)[nHits]) Hit(detID, pos, dpos, iPoint);
    new ((*mHits)[mNHits]) Hit(detID, pos, dpos, trackID);
    mNHits++;
  }

  LOG(INFO) << "Create " << mNHits << " hits out of "
            << nPoints << " points." << "";

  mTNofEvents++;
  mTNofHits += mNHits;

}

//_____________________________________________________________________________
void FindHits::execMQ(TList* inputList,TList* outputList) {

  LOG(INFO) << "FindHits::ExecMQ >>>>> (" << inputList->GetName() << "," << outputList->GetName() << "), Event " << mTNofEvents << "";

  mPoints = (TClonesArray*)inputList->FindObject("MFTPoints");

  outputList->Add(mHits);

  // use numbers from the MC event header ...
  mMCEventHeader = (FairMCEventHeader*)inputList->FindObject("MCEventHeader.");
  mEventHeader->SetRunId(mMCEventHeader->GetRunID());
  mEventHeader->SetMCEntryNumber(mMCEventHeader->GetEventID());
  mEventHeader->setPartNo(mMCEventHeader->GetNPrim());
  LOG(INFO) << "FindHits::ExecMQ >>>>> RunID " << mMCEventHeader->GetRunID() << " EventID " << mMCEventHeader->GetEventID() << " NPrim " << mMCEventHeader->GetNPrim() << "";
  outputList->Add(mEventHeader);

  Exec("");

  return;

}

//_____________________________________________________________________________
void FindHits::reset() 
{

  mNHits = 0;
  if (mHits) mHits->Clear();

}
