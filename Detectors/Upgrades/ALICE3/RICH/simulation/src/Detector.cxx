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

#include "DetectorsBase/Stack.h"
#include "ITSMFTSimulation/Hit.h"
#include "RICHSimulation/Detector.h"
#include "RICHBase/RICHBaseParam.h"

using o2::itsmft::Hit;

namespace o2
{
namespace rich
{

Detector::Detector()
  : o2::base::DetImpl<Detector>("RCH", true),
    mTrackData(),
    mHits(o2::utils::createSimVector<o2::itsmft::Hit>())
{
}

Detector::Detector(bool active)
  : o2::base::DetImpl<Detector>("RCH", true),
    mTrackData(),
    mHits(o2::utils::createSimVector<o2::itsmft::Hit>())
{
  auto& richPars = RICHBaseParam::Instance();
  mRings.resize(richPars.nRings);
  mNTiles = richPars.nTiles;
  LOGP(info, "Summary of RICH configuration:\n\tNumber of rings: {}\n\tNumber of tiles per ring: {}", mRings.size(), mNTiles);
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

  float tmaxfdSi = 0.1;    // .10000E+01; // Degree
  float stemaxSi = 0.0075; //  .10000E+01; // cm
  float deemaxSi = 0.1;    // 0.30000E-02; // Fraction of particle's energy 0<deemax<=1
  float epsilSi = 1.0E-4;  // .10000E+01;
  float stminSi = 0.0;     // cm "Default value used"

  float tmaxfdAir = 0.1;        // .10000E+01; // Degree
  float stemaxAir = .10000E+01; // cm
  float deemaxAir = 0.1;        // 0.30000E-02; // Fraction of particle's energy 0<deemax<=1
  float epsilAir = 1.0E-4;      // .10000E+01;
  float stminAir = 0.0;         // cm "Default value used"

  float tmaxfdCer = 0.1;        // .10000E+01; // Degree
  float stemaxCer = .10000E+01; // cm
  float deemaxCer = 0.1;        // 0.30000E-02; // Fraction of particle's energy 0<deemax<=1
  float epsilCer = 1.0E-4;      // .10000E+01;
  float stminCer = 0.0;         // cm "Default value used"

  float tmaxfdAerogel = 0.1;        // .10000E+01; // Degree
  float stemaxAerogel = .10000E+01; // cm
  float deemaxAerogel = 0.1;        // 0.30000E-02; // Fraction of particle's energy 0<deemax<=1
  float epsilAerogel = 1.0E-4;      // .10000E+01;
  float stminAerogel = 0.0;         // cm "Default value used"

  float tmaxfdArgon = 0.1;        // .10000E+01; // Degree
  float stemaxArgon = .10000E+01; // cm
  float deemaxArgon = 0.1;        // 0.30000E-02; // Fraction of particle's energy 0<deemax<=1
  float epsilArgon = 1.0E-4;      // .10000E+01;
  float stminArgon = 0.0;         // cm "Default value used"

  // AIR
  float aAir[4] = {12.0107, 14.0067, 15.9994, 39.948};
  float zAir[4] = {6., 7., 8., 18.};
  float wAir[4] = {0.000124, 0.755267, 0.231781, 0.012827};
  float dAir = 1.20479E-3;

  // Carbon fiber
  float aCf[2] = {12.0107, 1.00794};
  float zCf[2] = {6., 1.};

  // Silica aerogel https://pdg.lbl.gov/2023/AtomicNuclearProperties/HTML/silica_aerogel.html
  float aAerogel[3] = {15.9990, 28.0855, 1.00794};
  float zAerogel[3] = {8., 14., 1.};
  float wAerogel[3] = {0.543192, 0.453451, 0.003357};
  float dAerogel = 0.200; // g/cm3

  // Argon
  float aArgon = 39.948;
  float zArgon = 18.;
  float wArgon = 1.;
  float dArgon = 1.782E-3; // g/cm3

  o2::base::Detector::Mixture(1, "AIR$", aAir, zAir, dAir, 4, wAir);
  o2::base::Detector::Medium(1, "AIR$", 1, 0, ifield, fieldm, tmaxfdAir, stemaxAir, deemaxAir, epsilAir, stminAir);

  o2::base::Detector::Material(3, "SI$", 0.28086E+02, 0.14000E+02, 0.23300E+01, 0.93600E+01, 0.99900E+03);
  o2::base::Detector::Medium(3, "SI$", 3, 0, ifield, fieldm, tmaxfdSi, stemaxSi, deemaxSi, epsilSi, stminSi);

  o2::base::Detector::Mixture(2, "AEROGEL$", aAerogel, zAerogel, dAerogel, 3, wAerogel);
  o2::base::Detector::Medium(2, "AEROGEL$", 2, 0, ifield, fieldm, tmaxfdAerogel, stemaxAerogel, deemaxAerogel, epsilAerogel, stminAerogel);

  o2::base::Detector::Material(4, "ARGON$", aArgon, zArgon, dArgon, 1, wArgon);
  o2::base::Detector::Medium(4, "ARGON$", 4, 0, ifield, fieldm, tmaxfdArgon, stemaxArgon, deemaxArgon, epsilArgon, stminArgon);
}

void Detector::createGeometry()
{
  TGeoManager* geoManager = gGeoManager;
  TGeoVolume* vALIC = geoManager->GetVolume("barrel");
  if (!vALIC) {
    LOGP(fatal, "Could not find barrel volume while constructing RICH geometry");
  }
  new TGeoVolumeAssembly(GeometryTGeo::getRICHVolPattern());
  TGeoVolume* vRICH = geoManager->GetVolume(GeometryTGeo::getRICHVolPattern());
  vALIC->AddNode(vRICH, 2, new TGeoTranslation(0, 30., 0));

  char vstrng[100] = "RICHV";
  vRICH->SetTitle(vstrng);
  auto& richPars = RICHBaseParam::Instance();

  // TGeoTube* richVessel = new TGeoTube(richPars.rMin, richPars.rMax, richPars.zRichLength / 2.0);
  // TGeoMedium* medArgon = gGeoManager->GetMedium("RCH_ARGON$");
  // TGeoVolume* vRichVessel = new TGeoVolume(vstrng, richVessel, medArgon);
  // vRichVessel->SetLineColor(kGray);
  // vRichVessel->SetVisibility(kTRUE);
  // vRichVessel->SetTransparency(75);
  // vALIC->AddNode(vRichVessel, 1, new TGeoTranslation(0, 30., 0));

  if (!(richPars.nRings % 2)) {
    prepareEvenLayout();
  } else {
    prepareOddLayout();
  }
  for (int iRing{0}; iRing < richPars.nRings; ++iRing) {
    mRings[iRing] = Ring{iRing,
                         richPars.nTiles,
                         richPars.rMin,
                         richPars.rMax,
                         richPars.radiatorThickness,
                         (float)mVTile1[iRing],
                         (float)mVTile2[iRing],
                         (float)mLAerogelZ[iRing],
                         richPars.detectorThickness,
                         (float)mVMirror1[iRing],
                         (float)mVMirror2[iRing],
                         richPars.zBaseSize,
                         (float)mR0Radiator[iRing],
                         (float)mR0PhotoDet[iRing],
                         (float)mTRplusG[iRing],
                         (float)mThetaBi[iRing],
                         GeometryTGeo::getRICHVolPattern()};
  }
}

void Detector::InitializeO2Detector()
{
  LOG(info) << "Initialize RICH O2Detector";
  mGeometryTGeo = GeometryTGeo::Instance();
  // defineSensitiveVolumes();
}

void Detector::defineSensitiveVolumes()
{
  TGeoManager* geoManager = gGeoManager;
  TGeoVolume* v;

  TString volumeName;
  LOGP(info, "Adding RICH Sensitive Volumes");

  // The names of the RICH sensitive volumes have the format: Ring(0...mRings.size()-1)
  for (int j{0}; j < mRings.size(); j++) {
    volumeName = GeometryTGeo::getRICHSensorPattern() + TString::Itoa(j, 10);
    LOGP(info, "Trying {}", volumeName.Data());
    v = geoManager->GetVolume(volumeName.Data());
    LOGP(info, "Adding RICH Sensitive Volume {}", v->GetName());
    AddSensitiveVolume(v);
  }
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
  if (fMC->IsTrackExiting() && (lay == 0 || lay == mRings.size() - 1)) {
    // Keep the track refs for the innermost and outermost rings only
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

void Detector::prepareEvenLayout()
{
}

void Detector::prepareOddLayout()
{ // Mere translation of Nicola's code
  LOGP(info, "Setting up ODD layout for bRICH");
  auto& richPars = RICHBaseParam::Instance();

  mThetaBi.resize(richPars.nRings);
  mR0Tilt.resize(richPars.nRings);
  mZ0Tilt.resize(richPars.nRings);
  mLAerogelZ.resize(richPars.nRings);
  mTRplusG.resize(richPars.nRings);
  mMinRadialMirror.resize(richPars.nRings);
  mMaxRadialMirror.resize(richPars.nRings);
  mMaxRadialRadiator.resize(richPars.nRings);
  mVMirror1.resize(richPars.nRings);
  mVMirror2.resize(richPars.nRings);
  mVTile1.resize(richPars.nRings);
  mVTile2.resize(richPars.nRings);
  mR0Radiator.resize(richPars.nRings);
  mR0PhotoDet.resize(richPars.nRings);

  // Start from middle one
  double mVal = TMath::Tan(0.0);
  mThetaBi[richPars.nRings / 2] = TMath::ATan(mVal);
  mR0Tilt[richPars.nRings / 2] = richPars.rMax;
  mZ0Tilt[richPars.nRings / 2] = mR0Tilt[richPars.nRings / 2] * TMath::Tan(mThetaBi[richPars.nRings / 2]);
  mLAerogelZ[richPars.nRings / 2] = TMath::Sqrt(1.0 + mVal * mVal) * richPars.rMin * richPars.zBaseSize / (TMath::Sqrt(1.0 + mVal * mVal) * richPars.rMax - mVal * richPars.zBaseSize);
  mTRplusG[richPars.nRings / 2] = richPars.rMax - richPars.rMin;
  double t = TMath::Tan(TMath::ATan(mVal) + TMath::ATan(richPars.zBaseSize / (2.0 * richPars.rMax * TMath::Sqrt(1.0 + mVal * mVal) - richPars.zBaseSize * mVal)));
  mMinRadialMirror[richPars.nRings / 2] = richPars.rMax;
  mMaxRadialRadiator[richPars.nRings / 2] = richPars.rMin;

  // Configure rest of the rings
  for (int iRing{richPars.nRings / 2 + 1}; iRing < richPars.nRings; ++iRing) {
    double parA = t;
    double parB = 2.0 * richPars.rMax / richPars.zBaseSize;
    mVal = (TMath::Sqrt(parA * parA * parB * parB + parB * parB - 1.0) + parA * parB * parB) / (parB * parB - 1.0);
    t = TMath::Tan(TMath::ATan(mVal) + TMath::ATan(richPars.zBaseSize / (2.0 * richPars.rMax * TMath::Sqrt(1.0 + mVal * mVal) - richPars.zBaseSize * mVal)));
    // forward rings
    mThetaBi[iRing] = TMath::ATan(mVal);
    mR0Tilt[iRing] = richPars.rMax - richPars.zBaseSize / 2.0 * sin(TMath::ATan(mVal));
    mZ0Tilt[iRing] = mR0Tilt[iRing] * TMath::Tan(mThetaBi[iRing]);
    mLAerogelZ[iRing] = TMath::Sqrt(1.0 + mVal * mVal) * richPars.rMin * richPars.zBaseSize / (TMath::Sqrt(1.0 + mVal * mVal) * richPars.rMax - mVal * richPars.zBaseSize);
    mTRplusG[iRing] = TMath::Sqrt(1.0 + mVal * mVal) * (richPars.rMax - richPars.rMin) - mVal / 2.0 * (richPars.zBaseSize + mLAerogelZ[iRing]);
    mMinRadialMirror[iRing] = mR0Tilt[iRing] - richPars.zBaseSize / 2.0 * sin(TMath::ATan(mVal));
    mMaxRadialRadiator[iRing] = richPars.rMin + 2.0 * mLAerogelZ[iRing] / 2.0 * sin(TMath::ATan(mVal));
    // backward rings
    mThetaBi[2 * (richPars.nRings / 2) - iRing] = -TMath::ATan(mVal);
    mR0Tilt[2 * (richPars.nRings / 2) - iRing] = richPars.rMax - richPars.zBaseSize / 2.0 * sin(TMath::ATan(mVal));
    mZ0Tilt[2 * (richPars.nRings / 2) - iRing] = -mR0Tilt[iRing] * TMath::Tan(mThetaBi[iRing]);
    mLAerogelZ[2 * (richPars.nRings / 2) - iRing] = TMath::Sqrt(1.0 + mVal * mVal) * richPars.rMin * richPars.zBaseSize / (TMath::Sqrt(1.0 + mVal * mVal) * richPars.rMax - mVal * richPars.zBaseSize);
    mTRplusG[2 * (richPars.nRings / 2) - iRing] = TMath::Sqrt(1.0 + mVal * mVal) * (richPars.rMax - richPars.rMin) - mVal / 2.0 * (richPars.zBaseSize + mLAerogelZ[iRing]);
    mMinRadialMirror[2 * (richPars.nRings / 2) - iRing] = mR0Tilt[iRing] - richPars.zBaseSize / 2.0 * sin(TMath::ATan(mVal));
    mMaxRadialRadiator[2 * (richPars.nRings / 2) - iRing] = richPars.rMin + 2.0 * mLAerogelZ[iRing] / 2.0 * sin(TMath::ATan(mVal));
  }

  // Dimensioning tiles
  double percentage = 0.999;
  for (int iRing = 0; iRing < richPars.nRings; iRing++) {
    if (iRing == richPars.nRings / 2) {
      mVMirror1[iRing] = percentage * 2.0 * richPars.rMax * TMath::Sin(TMath::Pi() / double(richPars.nTiles));
      mVMirror2[iRing] = percentage * 2.0 * richPars.rMax * TMath::Sin(TMath::Pi() / double(richPars.nTiles));
      mVTile1[iRing] = percentage * 2.0 * richPars.rMin * TMath::Sin(TMath::Pi() / double(richPars.nTiles));
      mVTile2[iRing] = percentage * 2.0 * richPars.rMin * TMath::Sin(TMath::Pi() / double(richPars.nTiles));
    } else if (iRing > richPars.nRings / 2) {
      mVMirror1[iRing] = percentage * 2.0 * richPars.rMax * TMath::Sin(TMath::Pi() / double(richPars.nTiles));
      mVMirror2[iRing] = percentage * 2.0 * mMinRadialMirror[iRing] * TMath::Sin(TMath::Pi() / double(richPars.nTiles));
      mVTile1[iRing] = percentage * 2.0 * mMaxRadialRadiator[iRing] * TMath::Sin(TMath::Pi() / double(richPars.nTiles));
      mVTile2[iRing] = percentage * 2.0 * richPars.rMin * TMath::Sin(TMath::Pi() / double(richPars.nTiles));
    } else if (iRing < richPars.nRings / 2) {
      mVMirror2[iRing] = percentage * 2.0 * richPars.rMax * TMath::Sin(TMath::Pi() / double(richPars.nTiles));
      mVMirror1[iRing] = percentage * 2.0 * mMinRadialMirror[iRing] * TMath::Sin(TMath::Pi() / double(richPars.nTiles));
      mVTile2[iRing] = percentage * 2.0 * mMaxRadialRadiator[iRing] * TMath::Sin(TMath::Pi() / double(richPars.nTiles));
      mVTile1[iRing] = percentage * 2.0 * richPars.rMin * TMath::Sin(TMath::Pi() / double(richPars.nTiles));
    }
  }

  // Translation parameters
  for (size_t iRing{0}; iRing < richPars.nRings; ++iRing) {
    mR0Radiator[iRing] = mR0Tilt[iRing] - (mTRplusG[iRing] - richPars.radiatorThickness / 2) * TMath::Cos(mThetaBi[iRing]);
    mR0PhotoDet[iRing] = mR0Tilt[iRing] - (richPars.detectorThickness / 2) * TMath::Cos(mThetaBi[iRing]);
  }
}
} // namespace rich
} // namespace o2

ClassImp(o2::rich::Detector);