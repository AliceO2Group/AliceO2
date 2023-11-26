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

  TGeoTube* richVessel = new TGeoTube(richPars.rMin, richPars.rMax, richPars.zRichLength / 2.0);
  TGeoMedium* medArgon = gGeoManager->GetMedium("RCH_ARGON$");
  TGeoVolume* vRichVessel = new TGeoVolume(vstrng, richVessel, medArgon);
  vRichVessel->SetLineColor(kYellow);
  vRichVessel->SetVisibility(kTRUE);
  vRichVessel->SetTransparency(50);
  vALIC->AddNode(vRichVessel, 1, new TGeoTranslation(0, 30., 0));

  if (!(richPars.nRings % 2)) {
    prepareEvenLayout();
  } else {
    prepareOddLayout();
  }

  mRings[richPars.nRings / 2] = Ring{richPars.nRings,
                                     richPars.nTiles,
                                     richPars.rMin,
                                     richPars.rMax,
                                     richPars.radiatorThickness,
                                     mVTile1[richPars.nRings / 2],
                                     mVTile2[richPars.nRings / 2],
                                     mLAerogelZ[richPars.nRings / 2 / 2],
                                     richPars.detectorThickness,
                                     0,
                                     0,
                                     richPars.zBaseSize,
                                     GeometryTGeo::getRICHVolPattern()};
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
  // // Mere transcription of Nicola's code
  // auto& richPars = RICHBaseParam::Instance();
  // double twoHalvesGap = 1.0; // cm

  // int iCentralMirror = int((richPars.nRings) / 2.0);
  // double mVal = TMath::Tan(0.0);
  // double t = TMath::Tan(TMath::ATan(mVal) + TMath::ATan(twoHalvesGap / (2.0 * richPars.rMax * TMath::Sqrt(1.0 + mVal * mVal) - twoHalvesGap * mVal)));
  // for (int i = iCentralMirror; i < int(richPars.nRings); i++) {
  //   double parA = t;
  //   double parB = 2.0 * richPars.rMax / richPars.zBaseSize;
  //   // Equazione su wolphram
  //   // solve arctan(a) = arctan(x) - arctan(1/(b*TMath::Sqrt(1+x*x)-x))
  //   mVal = (TMath::Sqrt(parA * parA * parB * parB + parB * parB - 1.0) + parA * parB * parB) / (parB * parB - 1.0);
  //   thetaMin[i] = TMath::Pi() / 2.0 - TMath::ATan(t);
  //   thetaMax[2 * iCentralMirror - i - 1] = TMath::Pi() / 2.0 + TMath::ATan(t);
  //   t = TMath::Tan(TMath::ATan(mVal) + TMath::ATan(richPars.zBaseSize / (2.0 * richPars.rMax * TMath::Sqrt(1.0 + mVal * mVal) - richPars.zBaseSize * mVal)));
  //   thetaMax[i] = TMath::Pi() / 2.0 - TMath::ATan(t);
  //   thetaMin[2 * iCentralMirror - i - 1] = TMath::Pi() / 2.0 + TMath::ATan(t);
  //   // Avvaloro forward
  //   theta_bi[i] = TMath::ATan(mVal);
  //   r0tilt[i] = richPars.rMax - richPars.zBaseSize / 2.0 * sin(TMath::ATan(mVal));
  //   z0_tilt[i] = r0tilt[i] * TMath::Tan(theta_bi[i]);
  //   l_aerogel_z[i] = TMath::Sqrt(1.0 + mVal * mVal) * R_min * richPars.zBaseSize / (TMath::Sqrt(1.0 + mVal * mVal) * richPars.rMax - mVal * richPars.zBaseSize);
  //   T_r_plus_g[i] = TMath::Sqrt(1.0 + mVal * mVal) * (richPars.rMax - R_min) - mVal / 2.0 * (richPars.zBaseSize + l_aerogel_z[i]);
  //   min_radial_mirror[i] = r0tilt[i] - richPars.zBaseSize / 2.0 * sin(TMath::ATan(mVal));
  //   max_radial_radiator[i] = R_min + 2.0 * l_aerogel_z[i] / 2.0 * sin(TMath::ATan(mVal));
  //   // Avvaloro backword
  //   theta_bi[2 * iCentralMirror - i - 1] = -TMath::ATan(mVal);
  //   r0tilt[2 * iCentralMirror - i - 1] = richPars.rMax - richPars.zBaseSize / 2.0 * sin(TMath::ATan(mVal));
  //   z0_tilt[2 * iCentralMirror - i - 1] = -r0tilt[i] * TMath::Tan(theta_bi[i]);
  //   l_aerogel_z[2 * iCentralMirror - i - 1] = TMath::Sqrt(1.0 + mVal * mVal) * R_min * richPars.zBaseSize / (TMath::Sqrt(1.0 + mVal * mVal) * richPars.rMax - mVal * richPars.zBaseSize);
  //   T_r_plus_g[2 * iCentralMirror - i - 1] = TMath::Sqrt(1.0 + mVal * mVal) * (richPars.rMax - R_min) - mVal / 2.0 * (richPars.zBaseSize + l_aerogel_z[i]);
  //   min_radial_mirror[2 * iCentralMirror - i - 1] = r0tilt[i] - richPars.zBaseSize / 2.0 * sin(TMath::ATan(mVal));
  //   max_radial_radiator[2 * iCentralMirror - i - 1] = R_min + 2.0 * l_aerogel_z[i] / 2.0 * sin(TMath::ATan(mVal));
  // }

  // // Limiti coordinate in phi
  // double min_area_rad = 0.0;
  // double min_area_det = 0.0;
  // double percentage = 0.999;
  // for (int i = 0; i < int(richPars.nRings); i++) {
  //   if (i >= iCentralMirror) {
  //     v_mirror_1[i] = percentage * 2.0 * richPars.rMax * sin(TMath::Pi() / double(int(number_of_mirrors_in_rphi)));
  //     v_mirror_2[i] = percentage * 2.0 * min_radial_mirror[i] * sin(TMath::Pi() / double(int(number_of_mirrors_in_rphi)));
  //     v_tile_1[i] = percentage * 2.0 * max_radial_radiator[i] * sin(TMath::Pi() / double(int(number_of_mirrors_in_rphi)));
  //     v_tile_2[i] = percentage * 2.0 * R_min * sin(TMath::Pi() / double(int(number_of_mirrors_in_rphi)));
  //   } else if (i < iCentralMirror) {
  //     v_mirror_2[i] = percentage * 2.0 * richPars.rMax * sin(TMath::Pi() / double(int(number_of_mirrors_in_rphi)));
  //     v_mirror_1[i] = percentage * 2.0 * min_radial_mirror[i] * sin(TMath::Pi() / double(int(number_of_mirrors_in_rphi)));
  //     v_tile_2[i] = percentage * 2.0 * max_radial_radiator[i] * sin(TMath::Pi() / double(int(number_of_mirrors_in_rphi)));
  //     v_tile_1[i] = percentage * 2.0 * R_min * sin(TMath::Pi() / double(int(number_of_mirrors_in_rphi)));
  //   }
  //   min_area_rad += (v_tile_1[i] + v_tile_2[i]) * l_aerogel_z[i] / 2.0;
  //   min_area_det += (v_mirror_1[i] + v_mirror_2[i]) * richPars.zBaseSize / 2.0;
  //   if (i >= iCentralMirror && v_mirror_1[i] > *square_size_rphi)
  //     *square_size_rphi = v_mirror_1[i];
  // }
}

void Detector::prepareOddLayout()
{ // Mere transcription of Nicola's code
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

  // Start from middle one
  double mVal = TMath::Tan(0.0);
  mThetaBi[richPars.nRings / 2] = TMath::ATan(mVal);
  mR0Tilt[richPars.nRings / 2] = richPars.rMax;
  mZ0Tilt[richPars.nRings / 2] = mR0Tilt[richPars.nRings / 2] * TMath::Tan(mThetaBi[richPars.nRings / 2]);
  mLAerogelZ[richPars.nRings / 2] = TMath::Sqrt(1.0 + mVal * mVal) * richPars.rMin * richPars.zBaseSize / (TMath::Sqrt(1.0 + mVal * mVal) * richPars.rMax - mVal * richPars.zBaseSize);
  mTRplusG[richPars.nRings / 2] = richPars.rMax - richPars.rMin;
  double t = TMath::Tan(TMath::ATan(mVal) + TMath::ATan(richPars.zBaseSize / (2.0 * richPars.rMax * TMath::Sqrt(1.0 + mVal * mVal) - richPars.zBaseSize * mVal)));
  mMinRadialMirror[richPars.nRings / 2] = richPars.rMax;
  mMaxRadialMirror[richPars.nRings / 2] = richPars.rMin;

  // Configure rest of the rings
  for (auto iRing{richPars.nRings / 2 + 1}; iRing < richPars.nRings; ++iRing) {
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
    mMaxRadialMirror[iRing] = richPars.rMin + 2.0 * mLAerogelZ[iRing] / 2.0 * sin(TMath::ATan(mVal));
    // backward rings
    mThetaBi[2 * richPars.nRings / 2 - iRing] = -TMath::ATan(mVal);
    mR0Tilt[2 * richPars.nRings / 2 - iRing] = richPars.rMax - richPars.zBaseSize / 2.0 * sin(TMath::ATan(mVal));
    mZ0Tilt[2 * richPars.nRings / 2 - iRing] = -mR0Tilt[iRing] * TMath::Tan(mThetaBi[iRing]);
    mLAerogelZ[2 * richPars.nRings / 2 - iRing] = TMath::Sqrt(1.0 + mVal * mVal) * richPars.rMin * richPars.zBaseSize / (TMath::Sqrt(1.0 + mVal * mVal) * richPars.rMax - mVal * richPars.zBaseSize);
    mTRplusG[2 * richPars.nRings / 2 - iRing] = TMath::Sqrt(1.0 + mVal * mVal) * (richPars.rMax - richPars.rMin) - mVal / 2.0 * (richPars.zBaseSize + mLAerogelZ[iRing]);
    mMinRadialMirror[2 * richPars.nRings / 2 - iRing] = mR0Tilt[iRing] - richPars.zBaseSize / 2.0 * sin(TMath::ATan(mVal));
    mMaxRadialMirror[2 * richPars.nRings / 2 - iRing] = richPars.rMin + 2.0 * mLAerogelZ[iRing] / 2.0 * sin(TMath::ATan(mVal));
  }

  // Dimensioning tiles
  double percentage = 0.999;
  for (int iRing = 0; iRing < richPars.nRings; iRing++) {
    if (iRing == richPars.nRings / 2) {
      mVMirror1[iRing] = percentage * 2.0 * richPars.rMax * TMath::Sin(TMath::Pi() / double(richPars.nRings));
      mVMirror2[iRing] = percentage * 2.0 * richPars.rMax * TMath::Sin(TMath::Pi() / double(richPars.nRings));
      mVTile1[iRing] = percentage * 2.0 * richPars.rMin * TMath::Sin(TMath::Pi() / double(richPars.nRings));
      mVTile2[iRing] = percentage * 2.0 * richPars.rMin * TMath::Sin(TMath::Pi() / double(richPars.nRings));
    } else if (iRing > richPars.nRings / 2) {
      mVMirror1[iRing] = percentage * 2.0 * richPars.rMax * TMath::Sin(TMath::Pi() / double(richPars.nRings));
      mVMirror2[iRing] = percentage * 2.0 * mMinRadialMirror[iRing] * TMath::Sin(TMath::Pi() / double(richPars.nRings));
      mVTile1[iRing] = percentage * 2.0 * mMaxRadialRadiator[iRing] * TMath::Sin(TMath::Pi() / double(richPars.nRings));
      mVTile2[iRing] = percentage * 2.0 * richPars.rMin * TMath::Sin(TMath::Pi() / double(richPars.nRings));
    } else if (iRing < richPars.nRings / 2) {
      mVMirror2[iRing] = percentage * 2.0 * richPars.rMax * TMath::Sin(TMath::Pi() / double(richPars.nRings));
      mVMirror1[iRing] = percentage * 2.0 * mMinRadialMirror[iRing] * TMath::Sin(TMath::Pi() / double(richPars.nRings));
      mVTile2[iRing] = percentage * 2.0 * mMaxRadialRadiator[iRing] * TMath::Sin(TMath::Pi() / double(richPars.nRings));
      mVTile1[iRing] = percentage * 2.0 * richPars.rMin * TMath::Sin(TMath::Pi() / double(richPars.nRings));
    }
  }
}
} // namespace rich
} // namespace o2

ClassImp(o2::rich::Detector);