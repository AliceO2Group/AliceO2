// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See https://alice-o2.web.cern.ch/ for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file FindClusters.h
/// \brief Cluster finding from digits
/// \author bogdan.vulpescu@cern.ch 
/// \date 03/05/2017

#include "ITSMFTBase/Digit.h"

#include "MFTBase/Constants.h"
#include "MFTBase/Geometry.h"
#include "MFTSimulation/EventHeader.h"
#include "MFTReconstruction/Cluster.h"
#include "MFTReconstruction/ClusterizerTask.h"

#include "TClonesArray.h"
#include "TMath.h"

#include <FairMQLogger.h>
#include "FairMCEventHeader.h"

using namespace o2::MFT;

ClassImp(o2::MFT::ClusterizerTask)

//_____________________________________________________________________________
ClusterizerTask::ClusterizerTask():
mDigits(nullptr),
mClusters(nullptr),
mNClusters(0),
mTNofEvents(0),
mTNofClusters(0),
mMCEventHeader(nullptr),
mEventHeader(nullptr)
{

}

//_____________________________________________________________________________
ClusterizerTask::~ClusterizerTask()
{

  reset();
  if (mClusters) {
    mClusters->Delete();
    delete mClusters;
  }

}

//_____________________________________________________________________________
InitStatus ClusterizerTask::Init()
{

  LOG(INFO) << "ClusterizerTask::Init >>>>" << "";

  // Get RootManager
  FairRootManager* man = FairRootManager::Instance();
  if (!man) {
    //LOG(FATAL) << "RootManager not instantiated!" << "";
    return kFATAL;
  }

  // Get input array
  mDigits = static_cast<TClonesArray*>(man->GetObject("MFTDigits"));
  if (!mDigits) {
    //LOG(FATAL) << "No digits array!" << "";
    return kFATAL;
  }

  // Create and register output array
  mClusters = new TClonesArray("o2::MFT::Cluster");
  man->Register("MFTClusters", "MFT", mClusters, kTRUE);

  mEventHeader = new EventHeader();
  mEventHeader->SetName("EventHeader.");
  man->Register("EventHeader.","EvtHeader", mEventHeader, kFALSE);

  return kSUCCESS;

}

//_____________________________________________________________________________
InitStatus ClusterizerTask::ReInit()
{

  LOG(DEBUG) << "Re-Initilization of ClusterizerTask" << "";

  return kSUCCESS;

}

//_____________________________________________________________________________
void ClusterizerTask::initMQ(TList* tempList) 
{

  LOG(INFO) << "ClusterizerTask::InitMQ >>>>>" << "";

  mEventHeader = new EventHeader();
  mEventHeader->SetName("EventHeader.");
  mClusters = new TClonesArray("o2::MFT::Cluster",10000);

  return;

}

//_____________________________________________________________________________
void ClusterizerTask::Exec(Option_t* /*opt*/) 
{

  //Info("Exec","Exec called",0,0);
  LOG(INFO) << "ClusterizerTask::Exec >>>>>" << "";

  reset();
  /*
  o2::ITSMFT::Point *point;
  TVector3 pos, dpos;
  Int_t detID, trackID;
  Double_t dx = Geometry::sXPixelPitch/TMath::Sqrt(12);
  Double_t dy = Geometry::sYPixelPitch/TMath::Sqrt(12);
  Double_t dz = 0.;

  // Loop over fPoints
  Int_t nPoints = mPoints->GetEntriesFast();
  for (Int_t iPoint = 0; iPoint < nPoints; iPoint++) {
    point = static_cast<o2::ITSMFT::Hit*>(mPoints->At(iPoint));
    if (!point) continue;
    detID = point->GetDetectorID();
    trackID = point->GetTrackID();
    // copy the coordinates from point to hit
    pos.SetXYZ(point->GetStartX(),point->GetStartY(),point->GetStartZ());
    dpos.SetXYZ(dx,dy,dz);
    //new ((*fHits)[nHits]) Hit(detID, pos, dpos, iPoint);
    new ((*mHits)[mNClusters]) Hit(detID, pos, dpos, trackID);
    mNClusters++;
  }
  
  LOG(INFO) << "Create " << mNClusters << " clusters out of "
            << nDigits << " digits." << "";
  */
  mTNofEvents++;
  mTNofClusters += mNClusters;

}

//_____________________________________________________________________________
void ClusterizerTask::execMQ(TList* inputList,TList* outputList) {

  LOG(INFO) << "ClusterizerTask::ExecMQ >>>>> (" << inputList->GetName() << "," << outputList->GetName() << "), Event " << mTNofEvents << "";

  mDigits = (TClonesArray*)inputList->FindObject("MFTDigits");

  outputList->Add(mClusters);

  // use numbers from the MC event header ...
  mMCEventHeader = (FairMCEventHeader*)inputList->FindObject("MCEventHeader.");
  mEventHeader->SetRunId(mMCEventHeader->GetRunID());
  mEventHeader->SetMCEntryNumber(mMCEventHeader->GetEventID());
  mEventHeader->setPartNo(mMCEventHeader->GetNPrim());
  LOG(INFO) << "ClusterizerTask::ExecMQ >>>>> RunID " << mMCEventHeader->GetRunID() << " EventID " << mMCEventHeader->GetEventID() << " NPrim " << mMCEventHeader->GetNPrim() << "";
  outputList->Add(mEventHeader);

  Exec("");

  return;

}

//_____________________________________________________________________________
void ClusterizerTask::reset() 
{

  mNClusters = 0;
  if (mClusters) mClusters->Clear();

}
