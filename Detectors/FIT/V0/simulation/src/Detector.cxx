// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "TGeoManager.h" // for TGeoManager
#include "TMath.h"
#include "TGraph.h"
#include "TString.h"
#include "TSystem.h"
#include "TVirtualMC.h"
#include "TVector3.h"
#include "TLorentzVector.h"

#include "FairRootManager.h" // for FairRootManager
#include "FairLogger.h"
#include "FairVolume.h"

#include "FairRootManager.h"
#include "FairVolume.h"

#include <sstream>
#include "V0Simulation/Detector.h"
#include "V0Base/Geometry.h"
#include "SimulationDataFormat/Stack.h"

using namespace o2::v0;
using o2::v0::Geometry;

ClassImp(Detector);

Detector::Detector()
  : o2::Base::DetImpl<Detector>("V0", kTRUE),
    mHits(o2::utils::createSimVector<o2::v0::Hit>()),
    mGeometry(nullptr),
    mTrackData(){
  // Empty
}

Detector::~Detector() {
  if (mHits) {
    o2::utils::freeSimVector(mHits); // delete mHits;
  }
  if (mGeometry){
    delete mGeometry;
  }
}

Detector::Detector(Bool_t isActive)
  : o2::Base::DetImpl<Detector> ("V0", isActive),
    mHits(o2::utils::createSimVector<o2::v0::Hit>()),
    mGeometry(nullptr),
    mTrackData(){
  // Empty
}

void Detector::InitializeO2Detector()
{
  LOG(INFO) << "FIT_V0: Initializing O2 detector";

  TGeoVolume* volSensitive = gGeoManager->GetVolume("V0cell");
  if (!volSensitive) {
    LOG(FATAL) << "Can't find FIT V0 sensitive volume: cell";
  }
  else {
    AddSensitiveVolume(volSensitive);
    LOG(INFO) << "FIT-V0: Sensitive volume: " << volSensitive->GetName() << "   " << volSensitive->GetNumber();
    // TODO: Code from MFT
//    if (!mftGeom->getSensorVolumeID()) {
//      mftGeom->setSensorVolumeID(vol->GetNumber());
//    }
//    else if (mftGeom->getSensorVolumeID() != vol->GetNumber()) {
//      LOG(FATAL) << "CreateSensors: different Sensor volume ID !!!!";
//    }
  }
}

// TODO: Check if it works and remove some fields if same in MFT base as in T0
Bool_t Detector::ProcessHits(FairVolume* v){
  // This method is called from the MC stepping

  // Track only charged particles and photons
  bool isPhotonTrack = false;
  Int_t particleId = fMC->TrackPid();
  if (particleId == 50000050){ // If particle is photon
    isPhotonTrack = true;
  }
  if (!(isPhotonTrack || fMC->TrackCharge())) {
    return kFALSE;
  }

  // TODO: Uncomment or change the approach after geometry is ready
//  Geometry* v0Geo = Geometry::instance();
//  Int_t copy;
  // Check if hit is into a FIT-V0 sensitive volume
//  if (fMC->CurrentVolID(copy) != v0Geo->getSensorVolumeID())
//    return kFALSE;

  // Get unique ID of the cell
  Int_t cellId = -1;
  fMC->CurrentVolOffID(1, cellId);

  // Check track status to define when hit is started and when it is stopped
  bool startHit = false, stopHit = false;
  if ((fMC->IsTrackEntering()) || (fMC->IsTrackInside() && !mTrackData.mHitStarted)) {
    startHit = true;
  } else if ((fMC->IsTrackExiting() || fMC->IsTrackOut() || fMC->IsTrackStop())) {
    stopHit = true;
  }

  // Track is entering or created in the volume
  if (startHit) {
    mTrackData.mEnergyLoss = 0.;
    fMC->TrackMomentum(mTrackData.mMomentumStart);
    fMC->TrackPosition(mTrackData.mPositionStart);
    mTrackData.mHitStarted = true;
  }
  // Track is exiting or stopped within the volume
  if (stopHit) {
    TLorentzVector positionStop;
    fMC->TrackPosition(positionStop);
    Int_t trackID = fMC->GetStack()->GetCurrentTrackNumber();

    // TODO: compare this base with methods used by T0 (3 lines below)
    float etot = fMC->Etot();
    float eDep = fMC->Edep();
    addHit(trackID, cellId, particleId,
        mTrackData.mPositionStart.Vect(), positionStop.Vect(),
        mTrackData.mMomentumStart.Vect(), mTrackData.mMomentumStart.E(),
        positionStop.T(), mTrackData.mEnergyLoss, etot, eDep);
  }
  return kTRUE;
}

o2::v0::Hit* Detector::addHit(Int_t trackId, Int_t cellId, Int_t particleId,
    TVector3 startPos, TVector3 endPos,
    TVector3 startMom, double startE,
    double endTime, double eLoss, float eTot, float eDep){

  mHits->emplace_back(trackId, cellId, startPos, endPos, startMom, startE, endTime, eLoss, eTot, eDep);
  auto stack = (o2::Data::Stack*)fMC->GetStack();
  stack->addHit(GetDetId());
  return &(mHits->back());
}

// TODO: -> verify Todos inside the function
void Detector::createMaterials()
{
  LOG(INFO) << "FIT_V0: Creating materials";
  // Air mixture
  const Int_t nAir = 4;
  Float_t aAir[nAir] = { 12.0107,  14.0067,  15.9994,  39.948 };
  Float_t zAir[nAir] = { 6,        7,        8,        18 };
  Float_t wAir[nAir] = { 0.000124, 0.755267, 0.231781, 0.012827 };
  Float_t dAir = 0.00120479;

  // Scintillator mixture; TODO: Looks very rough, improve these numbers?
  const Int_t nScint = 2;
  Float_t aScint[nScint] = { 1,     12.01};
  Float_t zScint[nScint] = { 1,     6,   };
  Float_t wScint[nScint] = { 0.016, 0.984};
  Float_t dScint = 1.023;

  // Aluminum
//  Float_t aAlu = 26.98;
//  Float_t zAlu = 13.;
//  Float_t dAlu = 2.70;     // density [gr/cm^3]
//  Float_t radAlu = 8.897;  // rad len [cm]
//  Float_t absAlu = 39.70;  // abs len [cm]


  Int_t matId = 0;            // tmp material id number
  const Int_t unsens = 0, sens = 1; // sensitive or unsensitive medium

  // TODO: After the simulation is running cross run for both sets of numbers and verify if they matter to us -> choose faster solution
  Float_t tmaxfd = -10.0; // in t0: 10   // max deflection angle due to magnetic field in one step
  Float_t stemax = 0.001; // in t0: 0.1  // max step allowed [cm]
  Float_t deemax = -0.2;  // in t0: 1.0  // maximum fractional energy loss in one step 0<deemax<=1
  Float_t epsil = 0.001;  // in t0: 0.03 // tracking precision [cm]
  Float_t stmin = -0.001; // in t0: 0.03 // minimum step due to continuous processes [cm] (negative value: choose it automatically)

  Int_t fieldType;
  Float_t maxField;
  o2::Base::Detector::initFieldTrackingParams(fieldType, maxField);
  LOG(DEBUG) << "Detector::createMaterials >>>>> fieldType " << fieldType << " maxField " << maxField;

  // TODO: Comment out two lines below once tested that the above function assigns field type and max correctly
  fieldType = 2;
  maxField = 10.;

  o2::Base::Detector::Mixture(++matId, "Air$", aAir, zAir, dAir, nAir, wAir);
  o2::Base::Detector::Medium(Air, "Air$", matId, unsens, fieldType, maxField, tmaxfd, stemax, deemax, epsil, stmin);

  o2::Base::Detector::Mixture(++matId, "Scintillator$", aScint, zScint, dScint, nScint, wScint);
  o2::Base::Detector::Medium(Scintillator, "Scintillator$", matId, unsens, fieldType, maxField, tmaxfd, stemax, deemax, epsil, stmin);

//  o2::Base::Detector::Material(++matId, "Alu$", aAlu, zAlu, dAlu, radAlu, absAlu);
//  o2::Base::Detector::Medium(Alu, "Alu$", matId, unsens, fieldType, maxField, tmaxfd, stemax, deemax, epsil, stmin);

  LOG(DEBUG) << "Detector::createMaterials -----> matId = " << matId;
}

void Detector::ConstructGeometry()
{
  LOG(INFO) << "FIT_V0: Constructing geometry";
  createMaterials();
  mGeometry = new Geometry(Geometry::eOnlySensitive);
}

void Detector::Register()
{
  // This will create a branch in the output tree called Hit, setting the last
  // parameter to kFALSE means that this collection will not be written to the file,
  // it will exist only during the simulation

  if (FairRootManager::Instance()) {
    FairRootManager::Instance()->RegisterAny(addNameTo("Hit").data(), mHits, kTRUE);
  }
}

void Detector::EndOfEvent()
{
  Reset();
}

void Detector::Reset()
{
  if (!o2::utils::ShmManager::Instance().isOperational()) {
    mHits->clear();
  }
}
