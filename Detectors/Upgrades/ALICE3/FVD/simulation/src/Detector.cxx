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

/// \file Detector.cxx
/// \brief Implementation of the Detector class

#include "ITSMFTSimulation/Hit.h"
#include "FVDSimulation/Detector.h"
#include "FVDBase/GeometryTGeo.h"
#include "FVDBase/FVDBaseParam.h"

#include "DetectorsBase/Stack.h"
#include "SimulationDataFormat/TrackReference.h"
#include "Field/MagneticField.h"

// FairRoot includes
#include "FairDetector.h"
#include <fairlogger/Logger.h>
#include "FairRootManager.h"
#include "FairRun.h"
#include "FairRuntimeDb.h"
#include "FairVolume.h"
#include "FairRootManager.h"

#include "TVirtualMC.h"
#include "TLorentzVector.h"
#include "TVector3.h"
#include <TGeoTube.h>
#include <TGeoVolume.h>
#include <TGeoCompositeShape.h>
#include <TGeoMedium.h>
#include <TGeoCone.h>
#include <TGeoManager.h>
#include "TRandom.h"

class FairModule;

class TGeoMedium;

using namespace o2::fvd;
using o2::itsmft::Hit;

Detector::Detector(bool active)
  : o2::base::DetImpl<Detector>("FVD", true),
    mHits(o2::utils::createSimVector<o2::itsmft::Hit>()),
    mGeometryTGeo(nullptr),
    mTrackData()
{
  auto& baseParam = FVDBaseParam::Instance();
  mNumberOfSectors = baseParam.nsect;

  mDzScint = baseParam.dzscint/2;

  mRingRadiiA = baseParam.ringsA;
  mRingRadiiC = baseParam.ringsC;

  mNumberOfRingsA = mRingRadiiA.size() - 1;
  mNumberOfRingsC = mRingRadiiC.size() - 1;

  mZmodA = baseParam.zmodA;
  mZmodC = baseParam.zmodC;
}

Detector::Detector(const Detector& rhs)
  : o2::base::DetImpl<Detector>(rhs),
    mTrackData(),
    mHits(o2::utils::createSimVector<o2::itsmft::Hit>())
{
}

Detector& Detector::operator=(const Detector& rhs)
{

  if (this == &rhs) {
    return *this;
  }
  // base class assignment
  base::Detector::operator=(rhs);
  mTrackData = rhs.mTrackData;

  mHits = nullptr;
  return *this;
}

Detector::~Detector()
{

  if (mHits) {
    o2::utils::freeSimVector(mHits);
  }
}

void Detector::InitializeO2Detector()
{
  LOG(info) << "Initialize FVD detector";
  mGeometryTGeo = GeometryTGeo::Instance();
  defineSensitiveVolumes();
}

bool Detector::ProcessHits(FairVolume* vol)
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

  auto stack = (o2::data::Stack*)fMC->GetStack();

  //int cellId = vol->getVolumeId();

  // Check track status to define when hit is started and when it is stopped
  bool startHit = false, stopHit = false;
  unsigned char status = 0;

  Int_t currVolId, offId;

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
    return kFALSE; // do noting
  }

  if (startHit) {
    mTrackData.mHitStarted = true;
    mTrackData.mEnergyLoss = 0.;
    fMC->TrackMomentum(mTrackData.mMomentumStart);
    fMC->TrackPosition(mTrackData.mPositionStart);
    mTrackData.mTrkStatusStart = true;
  }

  if (stopHit) {
    TLorentzVector positionStop;
    fMC->TrackPosition(positionStop);
    int trackId = fMC->GetStack()->GetCurrentTrackNumber();

    int chId = getChannelId(mTrackData.mPositionStart.Vect());

    Hit* p = addHit(trackId, chId /*cellId*/, mTrackData.mPositionStart.Vect(), positionStop.Vect(),
                    mTrackData.mMomentumStart.Vect(), mTrackData.mMomentumStart.E(),
                    positionStop.T(), mTrackData.mEnergyLoss, mTrackData.mTrkStatusStart,
                    status);
    stack->addHit(GetDetId());
  } else {
    return false; // do nothing more
  }
  return true;
}

o2::itsmft::Hit* Detector::addHit(Int_t trackId, Int_t cellId,
                                  const TVector3& startPos,
                                  const TVector3& endPos,
                                  const TVector3& startMom,
                                  double startE,
                                  double endTime,
                                  double eLoss,
                                  unsigned int startStatus,
                                  unsigned int endStatus)
{
  mHits->emplace_back(trackId, cellId, startPos,
                      endPos, startMom, startE, endTime, eLoss, startStatus, endStatus);
  return &(mHits->back());
}

void Detector::ConstructGeometry()
{
  createMaterials();
  buildModules();
}

void Detector::EndOfEvent()
{
  Reset();
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

void Detector::createMaterials()
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
  Int_t fieldType = 3;              // Field type
  Float_t maxField = 5.0;           // Field max.

  Float_t tmaxfd = -10.0; // max deflection angle due to magnetic field in one step
  Float_t stemax = 0.1;   // max step allowed [cm]
  Float_t deemax = 1.0;   // maximum fractional energy loss in one step 0<deemax<=1
  Float_t epsil = 0.03;   // tracking precision [cm]
  Float_t stmin = -0.001; // minimum step due to continuous processes [cm] (negative value: choose it automatically)

  LOG(info) << "FVD: CreateMaterials(): fieldType " << fieldType << ", maxField " << maxField;

  o2::base::Detector::Mixture(++matId, "Scintillator", aScint, zScint, dScint, nScint, wScint);
  o2::base::Detector::Medium(Scintillator, "Scintillator", matId, unsens, fieldType, maxField,
                             tmaxfd, stemax, deemax, epsil, stmin);
}

void Detector::buildModules()
{
  LOGP(info, "Creating FVD geometry");

  TGeoVolume* vCave = gGeoManager->GetVolume("cave");

  if (!vCave) {
    LOG(fatal) << "Could not find the top volume (cave)!";
  }

  TGeoVolumeAssembly* vFVDA = buildModuleA();
  TGeoVolumeAssembly* vFVDC = buildModuleC();

  vCave->AddNode(vFVDA, 1, new TGeoTranslation(0., 0., mZmodA/* - mDzScint/2.*/));
  vCave->AddNode(vFVDC, 1, new TGeoTranslation(0., 0., mZmodC/* + mDzScint/2.*/));
}

TGeoVolumeAssembly* Detector::buildModuleA()
{
  TGeoVolumeAssembly* mod = new TGeoVolumeAssembly("FVDA");

  const TGeoMedium* medium = gGeoManager->GetMedium("FVD_Scintillator");

  float dphiDeg = 360. / mNumberOfSectors;

  for (int ir = 0; ir < mNumberOfRingsA; ir++) {
    std::string rName = "fvd_ring" + std::to_string(ir + 1);
    TGeoVolumeAssembly* ring = new TGeoVolumeAssembly(rName.c_str());
    for (int ic = 0; ic < mNumberOfSectors; ic++) {
      int cellId = ic + mNumberOfSectors * ir;
      std::string nodeName = "fvd_node" + std::to_string(cellId);
      float rmin = mRingRadiiA[ir];
      float rmax = mRingRadiiA[ir + 1];
      float phimin = dphiDeg * ic;
      float phimax = dphiDeg * (ic + 1);
      auto tbs = new TGeoTubeSeg("tbs", rmin, rmax, mDzScint, phimin, phimax);
      auto nod = new TGeoVolume(nodeName.c_str(), tbs, medium);
      nod->SetLineColor(kRed);
      ring->AddNode(nod, cellId);
    }
    mod->AddNode(ring, 1);
  }

  return mod;
}

TGeoVolumeAssembly* Detector::buildModuleC()
{
  TGeoVolumeAssembly* mod = new TGeoVolumeAssembly("FVDC");

  const TGeoMedium* medium = gGeoManager->GetMedium("FVD_Scintillator");

  float dphiDeg = 360. / mNumberOfSectors;

  for (int ir = 0; ir < mNumberOfRingsC; ir++) {
    std::string rName = "fvd_ring" + std::to_string(ir + 1 + mNumberOfRingsA);
    TGeoVolumeAssembly* ring = new TGeoVolumeAssembly(rName.c_str());
    for (int ic = 0; ic < mNumberOfSectors; ic++) {
      int cellId = ic + mNumberOfSectors * (ir + mNumberOfRingsA);
      std::string nodeName = "fvd_node" + std::to_string(cellId);
      float rmin = mRingRadiiC[ir];
      float rmax = mRingRadiiC[ir + 1];
      float phimin = dphiDeg * ic;
      float phimax = dphiDeg * (ic + 1);
      auto tbs = new TGeoTubeSeg("tbs", rmin, rmax, mDzScint, phimin, phimax);
      auto nod = new TGeoVolume(nodeName.c_str(), tbs, medium);
      nod->SetLineColor(kBlue);
      ring->AddNode(nod, cellId);
    }
    mod->AddNode(ring, 1);
  }

  return mod;
}

void Detector::defineSensitiveVolumes()
{
  LOG(info) << "Adding FVD Sentitive Volumes";

  TGeoVolume* v;
  TString volumeName;
  int nCellA = mNumberOfRingsA * mNumberOfSectors;
  int nCellC = mNumberOfRingsC * mNumberOfSectors;

  for (int iv = 0; iv < nCellA + nCellC; iv++) {
    volumeName = "fvd_node" + std::to_string(iv);
    v = gGeoManager->GetVolume(volumeName);
    LOG(info) << "Adding FVD Sensitive Volume => " << v->GetName();
    AddSensitiveVolume(v);
  }
  
}

int Detector::getChannelId(TVector3 vec)
{
  float phi = vec.Phi();
  if (phi < 0)
    phi += TMath::TwoPi();
  float r = vec.Perp();
  float z = vec.Z();

  int isect = int(phi / (TMath::Pi() / 4));

  std::vector<float> rd = z > 0 ? mRingRadiiA : mRingRadiiC;
  int noff = z > 0 ? 0 : mNumberOfRingsA * mNumberOfSectors;

  int ir = 0;

  for (int i = 1; i < rd.size(); i++) {
    if (r < rd[i])
      break;
    else
      ir++;
  }

  return ir * mNumberOfSectors + isect + noff;
}

ClassImp(o2::fvd::Detector);
