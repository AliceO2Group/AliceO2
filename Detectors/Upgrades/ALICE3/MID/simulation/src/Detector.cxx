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

#include <FairVolume.h>

#include <TVirtualMC.h>
#include <TVirtualMCStack.h>
#include <TGeoVolume.h>
#include <TGeoTube.h>
#include <TGeoManager.h>

#include "DetectorsBase/Stack.h"
#include "ITSMFTSimulation/Hit.h"
#include "MI3Simulation/Detector.h"
#include "MI3Base/MI3BaseParam.h"

using o2::itsmft::Hit;

namespace o2::mi3
{

Detector::Detector()
  : o2::base::DetImpl<Detector>("MI3", true),
    mTrackData(),
    mHits(o2::utils::createSimVector<o2::itsmft::Hit>())
{
}

Detector::Detector(bool active)
  : o2::base::DetImpl<Detector>("MI3", true),
    mTrackData(),
    mHits(o2::utils::createSimVector<o2::itsmft::Hit>())
{
  auto& midPars = MIDBaseParam::Instance();
}

Detector::~Detector()
{
  if (mHits) {
    o2::utils::freeSimVector(mHits);
  }
}

void Detector::ConstructGeometry()
{
  createMaterials();
  createGeometry();
}

void Detector::createMaterials()
{
  int ifield = 2;      // ?
  float fieldm = 10.0; // ?
  o2::base::Detector::initFieldTrackingParams(ifield, fieldm);

  float tmaxfdPolys = 0.1;        // .10000E+01; // Degree
  float stemaxPolys = .10000E+01; // cm
  float deemaxPolys = 0.1;        // 0.30000E-02; // Fraction of particle's energy 0<deemax<=1
  float epsilPolys = 1.0E-4;      // .10000E+01;
  float stminPolys = 0.0;         // cm "Default value used"

  // Materials
  // Air
  float aAir[4] = {12.0107, 14.0067, 15.9994, 39.948};
  float zAir[4] = {6., 7., 8., 18.};
  float wAir[4] = {0.000124, 0.755267, 0.231781, 0.012827};
  float dAir = 1.20479E-3;
  float dAir1 = 1.20479E-10;

  o2::base::Detector::Mixture(0, "AIR", aAir, zAir, dAir, 4, wAir);
  o2::base::Detector::Medium(0, "AIR", 0, 0, ifield, fieldm, tmaxfdPolys, stemaxPolys, deemaxPolys, epsilPolys, stminPolys);

  // Polystyrene (C6H5CHCH2)n https://pdg.lbl.gov/2023/AtomicNuclearProperties/HTML/polystyrene.html
  float aPolys[2] = {1.0080, 12.0107};
  float zPolys[2] = {1.f, 6};
  float wPolys[2] = {0.077418, 0.922582};
  float dPolys = 1.060; // g/cm3

  o2::base::Detector::Mixture(1, "POLYSTYRENE", aPolys, zPolys, dPolys, 2, wPolys);
  o2::base::Detector::Medium(1, "POLYSTYRENE", 1, 0, ifield, fieldm, tmaxfdPolys, stemaxPolys, deemaxPolys, epsilPolys, stminPolys);
}

void Detector::InitializeO2Detector()
{
  LOG(info) << "Initialize MID O2Detector";
  mGeometryTGeo = GeometryTGeo::Instance();
  // defineSensitiveVolumes();
}

void Detector::EndOfEvent() { Reset(); }

void Detector::Register()
{
  // This will create a branch in the output tree called Hit, setting the last
  // parameter to kFALSE means that this collection will not be written to the file,
  // it will exist only during the simulation

  if (FairRootManager::Instance()) {
    FairRootManager::Instance()->RegisterAny(addNameTo("Hit").data(), mHits, true);
  }
}

void Detector::createGeometry()
{
  LOGP(info, "Creating MID geometry");

  TGeoManager* geoManager = gGeoManager;
  TGeoVolume* vALIC = geoManager->GetVolume("barrel");
  if (!vALIC) {
    LOGP(fatal, "Could not find barrel volume while constructing MID geometry");
  }
  new TGeoVolumeAssembly(GeometryTGeo::getMIDVolPattern());
  TGeoVolume* vMID = geoManager->GetVolume(GeometryTGeo::getMIDVolPattern());
  vALIC->AddNode(vMID, 2, new TGeoTranslation(0, 30., 0));

  char vstrng[100] = "MIDVol";
  vMID->SetTitle(vstrng);

  // Build the MID
  mLayers.resize(2);
  mLayers[0] = MIDLayer(0, GeometryTGeo::composeSymNameLayer(0), 301.f, 500.f);
  mLayers[1] = MIDLayer(1, GeometryTGeo::composeSymNameLayer(1), 311.f, 520.f); // arbitrarily reduced to get multiple of 5.2f

  for (auto& layer : mLayers) {
    layer.createLayer(vMID);
  }
}

void Detector::Reset()
{
  if (!o2::utils::ShmManager::Instance().isOperational()) {
    mHits->clear();
  }
}

bool Detector::ProcessHits(FairVolume* vol)
{
  // This method is called from the MC stepping
  if (!(fMC->TrackCharge())) {
    return false;
  }

  int lay = vol->getVolumeId();
  int volID = vol->getMCid();

  // Is it needed to keep a track reference when the outer ITS volume is encountered?
  auto stack = (o2::data::Stack*)fMC->GetStack();
  if (fMC->IsTrackExiting() && (lay == 0)) {
    o2::TrackReference tr(*fMC, GetDetId());
    tr.setTrackID(stack->GetCurrentTrackNumber());
    tr.setUserId(lay);
    stack->addTrackReference(tr);
  }
  bool startHit = false, stopHit = false;
  unsigned char status = 0;
  if (fMC->IsTrackEntering()) {
    status |= Hit::kTrackEntering;
  }
  if (fMC->IsTrackInside()) {
    status |= Hit::kTrackInside;
  }
  if (fMC->IsTrackExiting()) {
    status |= Hit::kTrackExiting;
  }
  if (fMC->IsTrackOut()) {
    status |= Hit::kTrackOut;
  }
  if (fMC->IsTrackStop()) {
    status |= Hit::kTrackStopped;
  }
  if (fMC->IsTrackAlive()) {
    status |= Hit::kTrackAlive;
  }

  // track is entering or created in the volume
  if ((status & Hit::kTrackEntering) || (status & Hit::kTrackInside && !mTrackData.mHitStarted)) {
    startHit = true;
  } else if ((status & (Hit::kTrackExiting | Hit::kTrackOut | Hit::kTrackStopped))) {
    stopHit = true;
  }

  // increment energy loss at all steps except entrance
  if (!startHit) {
    mTrackData.mEnergyLoss += fMC->Edep();
  }
  if (!(startHit | stopHit)) {
    return false; // do noting
  }

  if (startHit) {
    mTrackData.mEnergyLoss = 0.;
    fMC->TrackMomentum(mTrackData.mMomentumStart);
    fMC->TrackPosition(mTrackData.mPositionStart);
    mTrackData.mTrkStatusStart = status;
    mTrackData.mHitStarted = true;
  }
  if (stopHit) {
    TLorentzVector positionStop;
    fMC->TrackPosition(positionStop);
    // Retrieve the indices with the volume path
    int stave(0), halfstave(0), chipinmodule(0), module;
    fMC->CurrentVolOffID(1, chipinmodule);
    fMC->CurrentVolOffID(2, module);
    fMC->CurrentVolOffID(3, halfstave);
    fMC->CurrentVolOffID(4, stave);

    Hit* p = addHit(stack->GetCurrentTrackNumber(), lay, mTrackData.mPositionStart.Vect(), positionStop.Vect(),
                    mTrackData.mMomentumStart.Vect(), mTrackData.mMomentumStart.E(), positionStop.T(),
                    mTrackData.mEnergyLoss, mTrackData.mTrkStatusStart, status);
    // p->SetTotalEnergy(vmc->Etot());

    // RS: not sure this is needed
    // Increment number of Detector det points in TParticle
    stack->addHit(GetDetId());
  }

  return true;
}

o2::itsmft::Hit* Detector::addHit(int trackID, int detID, const TVector3& startPos, const TVector3& endPos,
                                  const TVector3& startMom, double startE, double endTime, double eLoss, unsigned char startStatus,
                                  unsigned char endStatus)
{
  mHits->emplace_back(trackID, detID, startPos, endPos, startMom, startE, endTime, eLoss, startStatus, endStatus);
  return &(mHits->back());
}
} // namespace o2::mi3
ClassImp(o2::mi3::Detector);