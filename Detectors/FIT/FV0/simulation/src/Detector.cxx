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

/// \file   Detector.cxx
/// \brief  Implementation of the FV0 detector class.
///
/// \author Maciej Slupecki, University of Jyvaskyla, Finland
/// \author Andreas Molander, University of Helsinki, Finland

#include <TGeoManager.h>
#include <TLorentzVector.h>
#include <TString.h>
#include <TVirtualMC.h>
#include <FairRootManager.h>
#include <FairVolume.h>
#include "FV0Simulation/Detector.h"
#include "Framework/Logger.h"
#include "DetectorsBase/Stack.h"
#include "FV0Base/Geometry.h"

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
  LOG(info) << "FV0: Initializing O2 detector. Adding sensitive volumes.";

  std::string volSensitiveName;
  for (int i = 0; i < mGeometry->getSensitiveVolumeNames().size(); i++) {
    volSensitiveName = mGeometry->getSensitiveVolumeNames().at(i);
    TGeoVolume* volSensitive = gGeoManager->GetVolume(volSensitiveName.c_str());
    if (!volSensitive) {
      LOG(fatal) << "FV0: Can't find sensitive volume " << volSensitiveName;
    } else {
      AddSensitiveVolume(volSensitive);
      LOG(info) << "FV0: Sensitive volume added: " << volSensitive->GetName();
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
  LOG(info) << "FV0: Creating materials";

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
  Float_t wScint[nScint] = {0.07085, 0.92915}; // based on EJ-204 datasheet: n_atoms/cm3
  const Float_t dScint = 1.023;

  // PMMA plastic mixture: (C5O2H8)n, same for plastic fiber support and for the fiber core
  //   Fiber cladding is different, but it comprises only 3% of the fiber volume, so it is not included
  const Int_t nPlast = 3;
  Float_t aPlast[nPlast] = {1.00784, 12.0107, 15.999};
  Float_t zPlast[nPlast] = {1, 6, 8};
  Float_t wPlast[nPlast] = {0.08054, 0.59985, 0.31961};
  const Float_t dPlast = 1.18;

  // Densities of fiber-equivalent material, for five radially-distributed density regions
  const Int_t nFiberRings = 5;
  Float_t dFiberRings[nFiberRings] = {0.035631, 0.059611, 0.074765, 0.079451, 0.054490};

  // Densities of fiber-equivalent material, for five density regions in front of the PMTs
  const Int_t nFiberPMTs = 5;
  Float_t dFiberPMTs[nFiberPMTs] = {0.109313, 0.217216, 0.364493, 1.373307, 1.406480};

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

  // Mixture of elements found in the PMTs
  const Int_t nPMT = 14;
  Float_t aPMT[nPMT] = {63.546, 65.38, 28.085, 15.999, 12.011, 1.008, 14.007, 55.845, 51.996, 10.81, 121.76, 132.91, 9.0122, 26.982};
  Float_t zPMT[nPMT] = {29, 30, 14, 8, 6, 1, 7, 26, 24, 5, 51, 55, 4, 13};
  Float_t wPMT[nPMT] = {0.07, 0.02, 0.14, 0.21, 0.11, 0.02, 0.02, 0.04, 0.01, 0.01, 0.00, 0.00, 0.01, 0.34};
  const Float_t dPMT = Geometry::getPmtDensity();

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
  LOG(info) << "FV0: createMaterials(): fieldType " << fieldType << ", maxField " << maxField;

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

  for (int i = 0; i < nFiberRings; i++) {
    o2::base::Detector::Mixture(++matId, Form("FiberRing%i$", i + 1), aPlast, zPlast, dFiberRings[i], nPlast, wPlast);
    o2::base::Detector::Medium(FiberRing1 + i, Form("FiberRing%i$", i + 1), matId, unsens, fieldType, maxField,
                               tmaxfd, stemax, deemax, epsil, stmin);
  }

  for (int i = 0; i < nFiberPMTs; i++) {
    o2::base::Detector::Mixture(++matId, Form("FiberPMT%i$", i + 1), aPlast, zPlast, dFiberPMTs[i], nPlast, wPlast);
    o2::base::Detector::Medium(FiberPMT1 + i, Form("FiberPMT%i$", i + 1), matId, unsens, fieldType, maxField,
                               tmaxfd, stemax, deemax, epsil, stmin);
  }

  o2::base::Detector::Material(++matId, "Aluminium$", aAlu, zAlu, dAlu, 8.9, 999);
  o2::base::Detector::Medium(Aluminium, "Aluminium$", matId, unsens, fieldType, maxField,
                             tmaxfd, stemax, deemax, epsil, stmin);

  o2::base::Detector::Mixture(++matId, "Titanium$", aTitanium, zTitanium, dTitanium, nTitanium, wTitanium);
  o2::base::Detector::Medium(Titanium, "Titanium$", matId, unsens, fieldType, maxField,
                             tmaxfd, stemax, deemax, epsil, stmin);

  o2::base::Detector::Mixture(++matId, "PMT$", aPMT, zPMT, dPMT, nPMT, wPMT);
  o2::base::Detector::Medium(PMT, "PMT$", matId, unsens, fieldType, maxField,
                             tmaxfd, stemax, deemax, epsil, stmin);

  LOG(debug) << "FV0 Detector::createMaterials(): matId = " << matId;
}

void Detector::ConstructGeometry()
{
  LOG(info) << "FV0: Constructing geometry";
  createMaterials();
  mGeometry = Geometry::instance(Geometry::eFull);
  // mGeometry->enableComponent(Geometry::eScintillator, false);
  // mGeometry->enableComponent(Geometry::ePlastics, false);
  // mGeometry->enableComponent(Geometry::eFibers, false);
  // mGeometry->enableComponent(Geometry::eScrews, false);
  // mGeometry->enableComponent(Geometry::eRods, false);
  // mGeometry->enableComponent(Geometry::eAluminiumContainer, false);
  mGeometry->buildGeometry();
}
void Detector::addAlignableVolumes() const
{
  //
  // Creates entries for alignable volumes associating the symbolic volume
  // name with the corresponding volume path.
  //
  //  First version (mainly ported from AliRoot)
  //

  LOG(info) << "FV0: Add alignable volumes";

  if (!gGeoManager) {
    LOG(fatal) << "TGeoManager doesn't exist !";
    return;
  }

  TString volPath, symName;
  for (auto& half : {"RIGHT_0", "LEFT_1"}) {
    volPath = Form("/cave_1/barrel_1/FV0_1/FV0%s", half);
    symName = Form("FV0%s", half);
    LOG(info) << "FV0: Add alignable volume: " << symName << ": " << volPath;
    if (!gGeoManager->SetAlignableEntry(symName.Data(), volPath.Data())) {
      LOG(fatal) << "FV0: Unable to set alignable entry! " << symName << ": " << volPath;
    }
  }
}

o2::fv0::Hit* Detector::addHit(Int_t trackId, Int_t cellId,
                               const math_utils::Point3D<float>& startPos, const math_utils::Point3D<float>& endPos,
                               const math_utils::Vector3D<float>& startMom, double startE,
                               double endTime, double eLoss, Int_t particlePdg)
{
  mHits->emplace_back(trackId, cellId, startPos, endPos, startMom, startE, endTime, eLoss, particlePdg);
  auto stack = (o2::data::Stack*)fMC->GetStack();
  stack->addHit(GetDetId());
  return &(mHits->back());
}
