#include "Detector.h"
#include <stddef.h>             // for NULL
#include "Data/DetectorList.h"  // for DetectorId::kAliTpc
#include "Data/Stack.h"         // for Stack
#include "FairRootManager.h"    // for FairRootManager
#include "FairVolume.h"         // for FairVolume
#include "Point.h"              // for Point
#include "TClonesArray.h"       // for TClonesArray
#include "TVirtualMC.h"         // for TVirtualMC, gMC
#include "TVirtualMCStack.h"    // for TVirtualMCStack

#include <iostream>
using std::cout;
using std::endl;

using namespace AliceO2::TPC;

Detector::Detector()
  : AliceO2::Base::Detector("Detector", kTRUE, kAliTpc),
    mTrackNumberID(-1),
    mVolumeID(-1),
    mPosition(),
    mMomentum(),
    mTime(-1.),
    mLength(-1.),
    mEnergyLoss(-1),
    mPointCollection(new TClonesArray("DetectorPoint"))
{
}

Detector::Detector(const char* name, Bool_t active)
  : AliceO2::Base::Detector(name, active, kAliTpc),
    mTrackNumberID(-1),
    mVolumeID(-1),
    mPosition(),
    mMomentum(),
    mTime(-1.),
    mLength(-1.),
    mEnergyLoss(-1),
    mPointCollection(new TClonesArray("Point"))
{
}

Detector::~Detector()
{
  if (mPointCollection) {
    mPointCollection->Delete();
    delete mPointCollection;
  }
}

void Detector::Initialize()
{
    AliceO2::Base::Detector::Initialize();
}

Bool_t  Detector::ProcessHits(FairVolume* vol)
{
  /** This method is called from the MC stepping */

  //Set parameters at entrance of volume. Reset ELoss.
  if ( gMC->IsTrackEntering() ) {
    mEnergyLoss  = 0.;
    mTime   = gMC->TrackTime() * 1.0e09;
    mLength = gMC->TrackLength();
    gMC->TrackPosition(mPosition);
    gMC->TrackMomentum(mMomentum);
  }

  // Sum energy loss for all steps in the active volume
  mEnergyLoss += gMC->Edep();

  // Create DetectorPoint at exit of active volume
  if ( gMC->IsTrackExiting()    ||
       gMC->IsTrackStop()       ||
       gMC->IsTrackDisappeared()   ) {
    mTrackNumberID  = gMC->GetStack()->GetCurrentTrackNumber();
    mVolumeID = vol->getMCid();
    if (mEnergyLoss == 0. ) { return kFALSE; }
    AddHit(mTrackNumberID, mVolumeID, TVector3(mPosition.X(),  mPosition.Y(),  mPosition.Z()),
           TVector3(mMomentum.Px(), mMomentum.Py(), mMomentum.Pz()), mTime, mLength,
           mEnergyLoss);

    // Increment number of Detector det points in TParticle
    AliceO2::Data::Stack* stack = (AliceO2::Data::Stack*)gMC->GetStack();
    stack->AddPoint(kAliTpc);
    
  }

  return kTRUE;
}

void Detector::EndOfEvent()
{

  mPointCollection->Clear();

}



void Detector::Register()
{

  /** This will create a branch in the output tree called
      DetectorPoint, setting the last parameter to kFALSE means:
      this collection will not be written to the file, it will exist
      only during the simulation.
  */

  FairRootManager::Instance()->Register("TPCPoint", "TPC",mPointCollection, kTRUE);

}


TClonesArray* Detector::GetCollection(Int_t iColl) const
{
  if (iColl == 0) { return mPointCollection; }
  else { return NULL; }
}

void Detector::Reset()
{
  mPointCollection->Clear();
}

void Detector::ConstructGeometry()
{
  /** If you are using the standard ASCII input for the geometry
      just copy this and use it for your detector, otherwise you can
      implement here you own way of constructing the geometry. */

}

Point* Detector::AddHit(Int_t trackID, Int_t detID,
                                      TVector3 pos, TVector3 mom,
                                      Double_t time, Double_t length,
                                      Double_t eLoss)
{
  TClonesArray& clref = *mPointCollection;
  Int_t size = clref.GetEntriesFast();
  return new(clref[size]) Point(trackID, detID, pos, mom,
         time, length, eLoss);
}

ClassImp(AliceO2::TPC::Detector)
