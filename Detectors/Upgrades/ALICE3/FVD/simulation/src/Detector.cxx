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

#include "DataFormatsFVD/Hit.h"

#include "FVDBase/GeometryTGeo.h"

#include "FVDSimulation/Detector.h"

#include "DetectorsBase/Stack.h"
#include "Field/MagneticField.h"

#include "TVirtualMC.h"
#include "TLorentzVector.h"
#include "TVector3.h"
#include "TGeoManager.h"
#include "TRandom.h"

#include <fairlogger/Logger.h>
#include "FairRootManager.h"
#include "FairVolume.h"
#include "FairRootManager.h"

using namespace o2::fvd;
using o2::fvd::GeometryTGeo;
using o2::fvd::Hit;

ClassImp(o2::fvd::Detector);

//_____________________________________________________________________________
Detector::Detector(Bool_t Active)
  : o2::base::DetImpl<Detector>("FVD", Active),
    mHits(o2::utils::createSimVector<o2::fvd::Hit>()),
    mGeometry(nullptr)
{
}

//_____________________________________________________________________________
Detector::Detector(const Detector& src)
  : o2::base::DetImpl<Detector>(src),
    mHits(o2::utils::createSimVector<o2::fvd::Hit>())
{
}

//_____________________________________________________________________________
Detector& Detector::operator=(const Detector& src)
{

  if (this == &src) {
    return *this;
  }
  // base class assignment
  base::Detector::operator=(src);

  mHits = nullptr;
  return *this;
}

//_____________________________________________________________________________
Detector::~Detector()
{

  if (mHits) {
    o2::utils::freeSimVector(mHits);
  }
}

//_____________________________________________________________________________
void Detector::InitializeO2Detector()
{
  LOG(info) << "FVD: Initializing O2 detector. Adding sensitive volumes.";

  TGeoVolume* vol;

  vol = gGeoManager->GetVolume("FVDA");
  if (!vol) {
    LOG(fatal) << "can't find volume FVDA";
  } else {
    AddSensitiveVolume(vol);
  }

  vol = gGeoManager->GetVolume("FVDC");
  if (!vol) {
    LOG(fatal) << "can't find volume FVDC";
  } else {
    AddSensitiveVolume(vol);
  }
}

//_____________________________________________________________________________
Bool_t Detector::ProcessHits(FairVolume* vol)
{
  // This method is called from the MC stepping

  // Track only charged particles and photons
  bool isPhotonTrack = false;
  Int_t particlePdg = fMC->TrackPid();
  if (particlePdg == 22) { // If particle is standard PDG photon
    isPhotonTrack = true;
  }
  if (!(isPhotonTrack || fMC->TrackCharge())) {
    return kFALSE;
  }

  // Check track status to define when hit is started and when it is stopped
  bool startHit = false, stopHit = false;
  if ((fMC->IsTrackEntering()) || (fMC->IsTrackInside() && !mTrackData.mHitStarted)) {
    startHit = true;
  } else if ((fMC->IsTrackExiting() || fMC->IsTrackOut() || fMC->IsTrackStop())) {
    stopHit = true;
  }

  // Increment energy loss at all steps except entrance
  if (!startHit) {
    mTrackData.mEnergyLoss += fMC->Edep();
  }

  // Track is entering or created in the volume
  // Start registering new hit, defined as the combination of all the steps from a given particle
  if (startHit) {
    mTrackData.mHitStarted = true;
    mTrackData.mEnergyLoss = 0.;
    fMC->TrackMomentum(mTrackData.mMomentumStart);
    fMC->TrackPosition(mTrackData.mPositionStart);
  }

  if (stopHit) {
    TLorentzVector positionStop;
    fMC->TrackPosition(positionStop);
    Int_t trackID = fMC->GetStack()->GetCurrentTrackNumber();

    // Get unique ID of the detector cell (sensitive volume)
    Int_t cellId = mGeometry->getCurrentCellId(fMC);

    math_utils::Point3D<float> posStart(mTrackData.mPositionStart.X(), mTrackData.mPositionStart.Y(), mTrackData.mPositionStart.Z());
    math_utils::Point3D<float> posStop(positionStop.X(), positionStop.Y(), positionStop.Z());
    math_utils::Vector3D<float> momStart(mTrackData.mMomentumStart.Px(), mTrackData.mMomentumStart.Py(), mTrackData.mMomentumStart.Pz());
    addHit(trackID, cellId, posStart, posStop, momStart,
           mTrackData.mMomentumStart.E(), positionStop.T(),
           mTrackData.mEnergyLoss, particlePdg);
  } else {
    return kFALSE; // do noting more
  }

 return kTRUE;
}

//_____________________________________________________________________________
o2::fvd::Hit* Detector::addHit(Int_t trackId, Int_t cellId,
                               const math_utils::Point3D<float>& startPos, const math_utils::Point3D<float>& endPos,
                               const math_utils::Vector3D<float>& startMom, double startE,
                               double endTime, double eLoss, Int_t particlePdg)
{
  mHits->emplace_back(trackId, cellId, startPos, endPos, startMom, startE, endTime, eLoss, particlePdg);
  auto stack = (o2::data::Stack*)fMC->GetStack();
  stack->addHit(GetDetId());
  return &(mHits->back());
}

//_____________________________________________________________________________
void Detector::CreateMaterials()
{

  Float_t density, as[11], zs[11], ws[11];
  Double_t radLength, absLength, a_ad, z_ad;
  Int_t id;

  // EJ-204 scintillator, based on polyvinyltoluene
  const Int_t nScint = 2;
  Float_t aScint[nScint] = {1.00784, 12.0107};
  Float_t zScint[nScint] = {1, 6};
  Float_t wScint[nScint] = {0.07085, 0.92915}; // based on EJ-204 datasheet: n_atoms/cm3
  const Float_t dScint = 1.023;

  Int_t matId = 0;                  // tmp material id number
  const Int_t unsens = 0, sens = 1; // sensitive or unsensitive medium
				    //
  Int_t fieldType = 3;     // Field type
  Float_t maxField = 5.0; // Field max.

  Float_t tmaxfd = -10.0; // max deflection angle due to magnetic field in one step
  Float_t stemax = 0.1;   // max step allowed [cm]
  Float_t deemax = 1.0;   // maximum fractional energy loss in one step 0<deemax<=1
  Float_t epsil = 0.03;   // tracking precision [cm]
  Float_t stmin = -0.001; // minimum step due to continuous processes [cm] (negative value: choose it automatically)
			  
  LOG(info) << "FVD: createMaterials(): fieldType " << fieldType << ", maxField " << maxField;

  o2::base::Detector::Mixture(++matId, "Scintillator$", aScint, zScint, dScint, nScint, wScint);
  o2::base::Detector::Medium(Scintillator, "Scintillator$", matId, unsens, fieldType, maxField,
                             tmaxfd, stemax, deemax, epsil, stmin);

}

//_____________________________________________________________________________
void Detector::ConstructGeometry()
{
  CreateMaterials();
  mGeometry = new GeometryTGeo();
  mGeometry->Build();
}
//_____________________________________________________________________________
void Detector::Register()
{
  // This will create a branch in the output tree called Hit, setting the last
  // parameter to kFALSE means that this collection will not be written to the file,
  // it will exist only during the simulation

  if (FairRootManager::Instance()) {
    FairRootManager::Instance()->RegisterAny(addNameTo("Hit").data(), mHits, kTRUE);
  }
}

//_____________________________________________________________________________
void Detector::Reset()
{
  if (!o2::utils::ShmManager::Instance().isOperational()) {
    mHits->clear();
  }
}
