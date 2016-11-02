/// \file FindHits.h
/// \brief Simple hits finding from the points
/// \author bogdan.vulpescu@cern.ch 
/// \date 10/10/2016

#include "MFTBase/Constants.h"
#include "MFTSimulation/Point.h"
#include "MFTReconstruction/Hit.h"
#include "MFTReconstruction/FindHits.h"

#include "TClonesArray.h"
#include "TMath.h"

#include "FairMQLogger.h"

using namespace AliceO2::MFT;

ClassImp(AliceO2::MFT::FindHits)

//_____________________________________________________________________________
FindHits::FindHits():
fPoints(NULL),
fHits(NULL),
fNHits(0),
fTNofEvents(0),
fTNofHits(0)
{

}

//_____________________________________________________________________________
FindHits::~FindHits()
{

  Reset();
  if (fHits) {
    fHits->Delete();
    delete fHits;
  }

}

//_____________________________________________________________________________
InitStatus FindHits::Init()
{

  LOG(INFO) << "FindHits::Init >>>>" << "";

  // Get RootManager
  FairRootManager* ioman = FairRootManager::Instance();
  if ( !ioman ) {
    //LOG(FATAL) << "RootManager not instantiated!" << "";
    return kFATAL;
  }

  // Get input array
  fPoints = static_cast<TClonesArray*>(ioman->GetObject("MFTPoints"));
  if ( !fPoints ) {
    //LOG(FATAL) << "No Point array!" << "";
    return kFATAL;
  }

  // Create and register output array
  fHits = new TClonesArray("AliceO2::MFT::Hit");
  ioman->Register("MFTHits", "MFT", fHits, kTRUE);

  return kSUCCESS;

}

//_____________________________________________________________________________
InitStatus FindHits::ReInit()
{

  LOG(DEBUG) << "Re-Initilization of FindHits" << "";

  return kSUCCESS;

}

//_____________________________________________________________________________
void FindHits::InitMQ(TList* tempList) 
{

  LOG(INFO) << "FindHits::InitMQ >>>>>" << "";

  fHits = new TClonesArray("AliceO2::MFT::Hit",10000);

  return;

}

//_____________________________________________________________________________
void FindHits::Exec(Option_t* /*opt*/) 
{

  //Info("Exec","Exec called",0,0);
  LOG(INFO) << "FindHits::Exec >>>>>" << "";

  Reset();

  AliceO2::MFT::Point *point;
  TVector3 pos, dpos;
  Int_t detID, trackID;
  Double_t dx = Constants::kXPixelPitch/TMath::Sqrt(12);
  Double_t dy = Constants::kYPixelPitch/TMath::Sqrt(12);
  Double_t dz = 0.;

  // Loop over fPoints
  Int_t nPoints = fPoints->GetEntriesFast();
  for (Int_t iPoint = 0; iPoint < nPoints; iPoint++) {
    point = static_cast<AliceO2::MFT::Point*>(fPoints->At(iPoint));
    if ( !point) continue;
    detID = point->GetDetectorID();
    trackID = point->GetTrackID();
    // copy the coordinates from point to hit
    pos.SetXYZ(point->GetX(),point->GetY(),point->GetZ());
    dpos.SetXYZ(dx,dy,dz);
    //new ((*fHits)[nHits]) Hit(detID, pos, dpos, iPoint);
    new ((*fHits)[fNHits]) Hit(detID, pos, dpos, trackID);
    fNHits++;
  }

  LOG(INFO) << "Create " << fNHits << " hits out of "
            << nPoints << " points." << "";

  fTNofEvents++;
  fTNofHits += fNHits;

}

//_____________________________________________________________________________
void FindHits::ExecMQ(TList* inputList,TList* outputList) {

  LOG(INFO) << "FindHits::ExecMQ >>>>> (" << inputList->GetName() << "," << outputList->GetName() << "), Event " << fTNofEvents << "";

  fPoints = (TClonesArray*) inputList->FindObject("MFTPoints");
  outputList->Add(fHits);

  Exec("");

  return;

}

//_____________________________________________________________________________
void FindHits::Reset() 
{

  fNHits = 0;
  if (fHits) fHits->Clear();

}
