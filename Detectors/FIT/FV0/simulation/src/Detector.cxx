// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file   Detector.cxx
/// \brief  Implementation of the FV0 detector class.
///
/// \author Maciej Slupecki, University of Jyvaskyla, Finland
/// \author Andreas Molander, University of Helsinki, Finland

#include "FV0Simulation/Detector.h"

#include <sstream>

#include <TGeoManager.h>
#include <TGraph.h>
#include <TLorentzVector.h>
#include <TMath.h>
#include <TString.h>
#include <TSystem.h>
#include <TVector3.h>
#include <TVirtualMC.h>

#include <FairLogger.h>
#include <FairRootManager.h>
#include <FairVolume.h>

#include "FV0Base/Geometry.h"
#include "SimulationDataFormat/Stack.h"

using namespace o2::fv0;
using o2::fv0::Geometry;

ClassImp(Detector);

Detector::Detector()
  : o2::base::DetImpl<Detector>("FV0", kTRUE),
    mHits(o2::utils::createSimVector<o2::fv0::Hit>()),
    mGeometry(nullptr),
    mTrackData()
{
  // Empty
}

Detector::~Detector()
{
  if (mHits) {
    o2::utils::freeSimVector(mHits); // delete mHits;
  }
  if (mGeometry) {
    delete mGeometry;
  }
}

Detector::Detector(Bool_t isActive)
  : o2::base::DetImpl<Detector>("FV0", isActive),
    mHits(o2::utils::createSimVector<o2::fv0::Hit>()),
    mGeometry(nullptr),
    mTrackData()
{
  // Empty
}

void Detector::InitializeO2Detector()
{
  LOG(INFO) << "FV0: Initializing O2 detector. Adding sensitive volumes.";

  std::string volSensitiveName;
  for (int i = 0; i < mGeometry->getSensitiveVolumeNames().size(); i++) {
    volSensitiveName = mGeometry->getSensitiveVolumeNames().at(i);
    TGeoVolume* volSensitive = gGeoManager->GetVolume(volSensitiveName.c_str());
    if (!volSensitive) {
      LOG(FATAL) << "FV0: Can't find sensitive volume " << volSensitiveName;
    } else {
      AddSensitiveVolume(volSensitive);
      LOG(INFO) << "FV0: Sensitive volume added: " << volSensitive->GetName();
    }
  }
}

Bool_t Detector::ProcessHits(FairVolume* v)
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

  // Track is exiting or stopped within the volume, finalize recording of this hit and save it
  if (stopHit) {
    TLorentzVector positionStop;
    fMC->TrackPosition(positionStop);
    Int_t trackID = fMC->GetStack()->GetCurrentTrackNumber();

    // Get unique ID of the detector cell (sensitive volume)
    Int_t cellId = mGeometry->getCurrentCellId(fMC);

    Point3D<float> posStart(mTrackData.mPositionStart.X(), mTrackData.mPositionStart.Y(), mTrackData.mPositionStart.Z());
    Point3D<float> posStop(positionStop.X(), positionStop.Y(), positionStop.Z());
    Vector3D<float> momStart(mTrackData.mMomentumStart.Px(), mTrackData.mMomentumStart.Py(), mTrackData.mMomentumStart.Pz());
    addHit(trackID, cellId, posStart, posStop, momStart,
           mTrackData.mMomentumStart.E(), positionStop.T(),
           mTrackData.mEnergyLoss, particlePdg);
  } else {
    return kFALSE; // do noting more
  }

  return kTRUE;
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

void Detector::Reset()
{
  if (!o2::utils::ShmManager::Instance().isOperational()) {
    mHits->clear();
  }
}

void Detector::EndOfEvent()
{
  Reset();
}

void Detector::createMaterials()
{
  LOG(INFO) << "FV0: Creating materials";

  // Air mixture
  const Int_t nAir = 4;
  Float_t aAir[nAir] = {12.0107, 14.0067, 15.9994, 39.948};
  Float_t zAir[nAir] = {6, 7, 8, 18};
  Float_t wAir[nAir] = {0.000124, 0.755267, 0.231781, 0.012827};
  const Float_t dAir = 0.00120479;

  // EJ-204 scintillator, based on polyvinyltoluene
  const Int_t nScint = 2;
  Float_t aScint[nScint] = {1.00784, 12.0107};
  Float_t zScint[nScint] = {1, 6};
  // TODO: Verify which of the following 2 lines is correct
  Float_t wScint[nScint] = {0.07085, 0.92915}; // based on EJ-204 datasheet: n_atoms/cm3
  //Float_t wScint[nScint] = { 0.08528, 0.91472 }; // based on chemical composition of base: polyvinyltoluene
  const Float_t dScint = 1.023;

  // PMMA plastic mixture: (C5O2H8)n, same for plastic fiber support and for the fiber core
  //   Fiber cladding is different, but it comprises only 3% of the fiber volume, so it is not included
  const Int_t nPlast = 3;
  Float_t aPlast[nPlast] = {1.00784, 12.0107, 15.999};
  Float_t zPlast[nPlast] = {1, 6, 8};
  Float_t wPlast[nPlast] = {0.08054, 0.59985, 0.31961};
  const Float_t dPlast = 1.18;

  // Densities of fiber-equivalent material, for 3 radially-distributed density regions
  Float_t dFiberInner = 0.087;
  Float_t dFiberMiddle = 0.129;
  Float_t dFiberOuter = 0.049;

  // Aluminum
  Float_t aAlu = 26.981;
  Float_t zAlu = 13;
  Float_t dAlu = 2.7;

  // Titanium grade 5 (https://en.wikipedia.org/wiki/Titanium_alloy) without iron and oxygen
  const Int_t nTitanium = 3;
  Float_t aTitanium[nTitanium] = {47.87, 26.98, 50.94};
  Float_t zTitanium[nTitanium] = {22, 13, 23};
  Float_t wTitanium[nTitanium] = {0.9, 0.06, 0.04};
  const Float_t dTitanium = 4.42;

  Int_t matId = 0;                  // tmp material id number
  const Int_t unsens = 0, sens = 1; // sensitive or unsensitive medium

  Float_t tmaxfd = -10.0; // max deflection angle due to magnetic field in one step
  Float_t stemax = 0.1;   // max step allowed [cm]
  Float_t deemax = 1.0;   // maximum fractional energy loss in one step 0<deemax<=1
  Float_t epsil = 0.03;   // tracking precision [cm]
  Float_t stmin = -0.001; // minimum step due to continuous processes [cm] (negative value: choose it automatically)

  Int_t fieldType;
  Float_t maxField;
  o2::base::Detector::initFieldTrackingParams(fieldType, maxField);
  LOG(INFO) << "FV0 Detector::createMaterials(): fieldType " << fieldType << ", maxField " << maxField;

  // TODO: Comment out two lines below once tested that the above function assigns field type and max correctly
  fieldType = 2;
  maxField = 10.;

  o2::base::Detector::Mixture(++matId, "Air$", aAir, zAir, dAir, nAir, wAir);
  o2::base::Detector::Medium(Air, "Air$", matId, unsens, fieldType, maxField,
                             tmaxfd, stemax, deemax, epsil, stmin);

  o2::base::Detector::Mixture(++matId, "Scintillator$", aScint, zScint, dScint, nScint, wScint);
  o2::base::Detector::Medium(Scintillator, "Scintillator$", matId, unsens, fieldType, maxField,
                             tmaxfd, stemax, deemax, epsil, stmin);

  o2::base::Detector::Mixture(++matId, "Plastic$", aPlast, zPlast, dPlast, nPlast, wPlast);
  o2::base::Detector::Medium(Plastic, "Plastic$", matId, unsens, fieldType, maxField,
                             tmaxfd, stemax, deemax, epsil, stmin);

  o2::base::Detector::Mixture(++matId, "FiberInner$", aPlast, zPlast, dFiberInner, nPlast, wPlast);
  o2::base::Detector::Medium(FiberInner, "FiberInner$", matId, unsens, fieldType, maxField,
                             tmaxfd, stemax, deemax, epsil, stmin);

  o2::base::Detector::Mixture(++matId, "FiberMiddle$", aPlast, zPlast, dFiberMiddle, nPlast, wPlast);
  o2::base::Detector::Medium(FiberMiddle, "FiberMiddle$", matId, unsens, fieldType, maxField,
                             tmaxfd, stemax, deemax, epsil, stmin);

  o2::base::Detector::Mixture(++matId, "FiberOuter$", aPlast, zPlast, dFiberOuter, nPlast, wPlast);
  o2::base::Detector::Medium(FiberOuter, "FiberOuter$", matId, unsens, fieldType, maxField,
                             tmaxfd, stemax, deemax, epsil, stmin);

  o2::base::Detector::Material(++matId, "Aluminium$", aAlu, zAlu, dAlu, 8.9, 999);
  o2::base::Detector::Medium(Aluminium, "Aluminium$", matId, unsens, fieldType, maxField,
                             tmaxfd, stemax, deemax, epsil, stmin);

  o2::base::Detector::Mixture(++matId, "Titanium$", aTitanium, zTitanium, dTitanium, nTitanium, wTitanium);
  o2::base::Detector::Medium(Titanium, "Titanium$", matId, unsens, fieldType, maxField,
                             tmaxfd, stemax, deemax, epsil, stmin);

  LOG(DEBUG) << "FV0 Detector::createMaterials(): matId = " << matId;
}

void Detector::ConstructGeometry()
{
  LOG(INFO) << "FV0: Constructing geometry";
  createMaterials();
  mGeometry = new Geometry(Geometry::eFull);
  // mGeometry->enableComponent(Geometry::eScintillator, false);
  // mGeometry->enableComponent(Geometry::ePlastics, false);
  // mGeometry->enableComponent(Geometry::eFibers, false);
  // mGeometry->enableComponent(Geometry::eScrews, false);
  // mGeometry->enableComponent(Geometry::eRods, false);
  // mGeometry->enableComponent(Geometry::eAluminiumContainer, false);
  mGeometry->buildGeometry();
}

o2::fv0::Hit* Detector::addHit(Int_t trackId, Int_t cellId,
                               const Point3D<float>& startPos, const Point3D<float>& endPos,
                               const Vector3D<float>& startMom, double startE,
                               double endTime, double eLoss, Int_t particlePdg)
{
  mHits->emplace_back(trackId, cellId, startPos, endPos, startMom, startE, endTime, eLoss, particlePdg);
  auto stack = (o2::data::Stack*)fMC->GetStack();
  stack->addHit(GetDetId());
  return &(mHits->back());
}
