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

#include "DataFormatsFD3/Hit.h"
#include "FD3Simulation/Detector.h"
#include "FD3Base/GeometryTGeo.h"
#include "FD3Base/FD3BaseParam.h"
#include "FD3Base/Constants.h"

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
#include <cmath>

class FairModule;

class TGeoMedium;

using namespace o2::fd3;
using o2::fd3::Hit;

Detector::Detector(bool active)
  : o2::base::DetImpl<Detector>("FD3", true),
    mHits(o2::utils::createSimVector<o2::fd3::Hit>()),
    mGeometryTGeo(nullptr),
    mTrackData()
{
  mNumberOfRingsC = Constants::nringsC;
  mNumberOfSectors = Constants::nsect;

  mEtaMinA = Constants::etaMin;
  mEtaMaxA = Constants::etaMax;
  mEtaMinC = -Constants::etaMax;
  mEtaMaxC = -Constants::etaMin;

  auto& baseParam = FD3BaseParam::Instance();

  if (baseParam.withMG) {
    mNumberOfRingsA = Constants::nringsA_withMG;
    mEtaMinA = Constants::etaMinA_withMG;
  } else {
    mNumberOfRingsA = Constants::nringsA;
    mEtaMinA = Constants::etaMin;
  }

  mDzScint = baseParam.dzscint / 2;
  mDzPlate = baseParam.dzplate;

  mPlateBehindA = baseParam.plateBehindA;
  mFullContainer = baseParam.fullContainer;

  mZA = baseParam.zmodA;
  mZC = baseParam.zmodC;

  for (int i = 0; i <= mNumberOfRingsA + 1; i++) {
    float eta = mEtaMaxA - i * (mEtaMaxA - mEtaMinA) / mNumberOfRingsA;
    float r = ringSize(mZA, eta);
    mRingSizesA.emplace_back(r);
  }

  for (int i = 0; i <= mNumberOfRingsC + 1; i++) {
    float eta = mEtaMinC + i * (mEtaMaxC - mEtaMinC) / mNumberOfRingsC;
    float r = ringSize(mZC, eta);
    mRingSizesC.emplace_back(r);
  }
}

Detector::Detector(const Detector& rhs)
  : o2::base::DetImpl<Detector>(rhs),
    mTrackData(),
    mHits(o2::utils::createSimVector<o2::fd3::Hit>())
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
  LOG(info) << "Initialize Forward Detector";
  mGeometryTGeo = GeometryTGeo::Instance();
  defineSensitiveVolumes();
}

bool Detector::ProcessHits(FairVolume* vol)
{
  // This method is called from the MC stepping
  if (!(fMC->TrackCharge())) {
    return kFALSE;
  }

  int detId;
  int volID = fMC->CurrentVolID(detId);

  auto stack = (o2::data::Stack*)fMC->GetStack();

  // Check track status to define when hit is started and when it is stopped
  int particlePdg = fMC->TrackPid();
  bool startHit = false, stopHit = false;
  if ((fMC->IsTrackEntering()) || (fMC->IsTrackInside() && !mTrackData.mHitStarted)) {
    startHit = true;
  } else if ((fMC->IsTrackExiting() || fMC->IsTrackOut() || fMC->IsTrackStop())) {
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
    int trackId = stack->GetCurrentTrackNumber();

    math_utils::Point3D<float> posStart(mTrackData.mPositionStart.X(), mTrackData.mPositionStart.Y(), mTrackData.mPositionStart.Z());
    math_utils::Point3D<float> posStop(positionStop.X(), positionStop.Y(), positionStop.Z());
    math_utils::Vector3D<float> momStart(mTrackData.mMomentumStart.Px(), mTrackData.mMomentumStart.Py(), mTrackData.mMomentumStart.Pz());

    Hit* p = addHit(trackId, detId, posStart, posStop,
                    momStart, mTrackData.mMomentumStart.E(),
                    positionStop.T(), mTrackData.mEnergyLoss, particlePdg);
    stack->addHit(GetDetId());
  } else {
    return false; // do nothing more
  }
  return true;
}

o2::fd3::Hit* Detector::addHit(int trackId, unsigned int detId,
                               const math_utils::Point3D<float>& startPos,
                               const math_utils::Point3D<float>& endPos,
                               const math_utils::Vector3D<float>& startMom,
                               double startE,
                               double endTime,
                               double eLoss,
                               int particlePdg)
{
  mHits->emplace_back(trackId, detId, startPos,
                      endPos, startMom, startE, endTime, eLoss, particlePdg);
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
  float density, as[11], zs[11], ws[11];
  double radLength, absLength, a_ad, z_ad;
  int id;

  // EJ-204 scintillator, based on polyvinyltoluene
  const int nScint = 2;
  float aScint[nScint] = {1.00784, 12.0107};
  float zScint[nScint] = {1, 6};
  float wScint[nScint] = {0.07085, 0.92915}; // based on EJ-204 datasheet: n_atoms/cm3
  const float dScint = 1.023;

  // Aluminium
  Float_t aAlu = 26.981;
  Float_t zAlu = 13;
  Float_t dAlu = 2.7;

  int matId = 0;                  // tmp material id number
  const int unsens = 0, sens = 1; // sensitive or unsensitive medium
                                  //
  int fieldType = 3;              // Field type
  float maxField = 5.0;           // Field max.

  float tmaxfd3 = -10.0; // max deflection angle due to magnetic field in one step
  float stemax = 0.1;    // max step allowed [cm]
  float deemax = 1.0;    // maximum fractional energy loss in one step 0<deemax<=1
  float epsil = 0.03;    // tracking precision [cm]
  float stmin = -0.001;  // minimum step due to continuous processes [cm] (negative value: choose it automatically)

  LOG(info) << "FD3: CreateMaterials(): fieldType " << fieldType << ", maxField " << maxField;

  o2::base::Detector::Mixture(++matId, "Scintillator", aScint, zScint, dScint, nScint, wScint);
  o2::base::Detector::Medium(Scintillator, "Scintillator", matId, unsens, fieldType, maxField,
                             tmaxfd3, stemax, deemax, epsil, stmin);

  o2::base::Detector::Material(++matId, "Aluminium", aAlu, zAlu, dAlu, 8.9, 999);
  o2::base::Detector::Medium(Aluminium, "Aluminium", matId, unsens, fieldType, maxField,
                             tmaxfd3, stemax, deemax, epsil, stmin);
}

void Detector::buildModules()
{
  LOGP(info, "Creating FD3 geometry");

  TGeoVolume* vCave = gGeoManager->GetVolume("cave");

  if (!vCave) {
    LOG(fatal) << "Could not find the top volume!";
  }

  TGeoVolumeAssembly* vFD3A = buildModuleA();
  TGeoVolumeAssembly* vFD3C = buildModuleC();

  vCave->AddNode(vFD3A, 1, new TGeoTranslation(0., 0., mZA));
  vCave->AddNode(vFD3C, 2, new TGeoTranslation(0., 0., mZC));
}

TGeoVolumeAssembly* Detector::buildModuleA()
{
  TGeoVolumeAssembly* mod = new TGeoVolumeAssembly("FD3A");

  const TGeoMedium* medium = gGeoManager->GetMedium("FD3_Scintillator");

  float dphiDeg = 360. / mNumberOfSectors;

  for (int ir = 0; ir < mNumberOfRingsA; ir++) {
    std::string rName = "fd3_ring" + std::to_string(ir + 1);
    TGeoVolumeAssembly* ring = new TGeoVolumeAssembly(rName.c_str());
    float rmin = mRingSizesA[ir];
    float rmax = mRingSizesA[ir + 1];
    LOG(info) << "ring" << ir << ": from " << rmin << " to " << rmax;
    for (int ic = 0; ic < mNumberOfSectors; ic++) {
      int cellId = ic + mNumberOfSectors * ir;
      std::string nodeName = "fd3_node" + std::to_string(cellId);
      float phimin = dphiDeg * ic;
      float phimax = dphiDeg * (ic + 1);
      auto tbs = new TGeoTubeSeg("tbs", rmin, rmax, mDzScint, phimin, phimax);
      auto nod = new TGeoVolume(nodeName.c_str(), tbs, medium);
      if ((ir + ic) % 2 == 0) {
        nod->SetLineColor(kRed);
      } else {
        nod->SetLineColor(kRed - 7);
      }
      ring->AddNode(nod, cellId);
    }
    mod->AddNode(ring, ir + 1);
  }

  // Aluminium plates on one or both sides of the A side module
  if (mPlateBehindA || mFullContainer) {
    LOG(info) << "adding container on A side";
    auto pmed = (TGeoMedium*)gGeoManager->GetMedium("FD3_Aluminium");
    auto pvol = new TGeoTube("pvol_fd3a", mRingSizesA[0], mRingSizesA[mNumberOfRingsA], mDzPlate);
    auto pnod1 = new TGeoVolume("pnod1_FD3A", pvol, pmed);
    double dpz = 2. + mDzPlate / 2;
    mod->AddNode(pnod1, 1, new TGeoTranslation(0, 0, dpz));

    if (mFullContainer) {
      auto pnod2 = new TGeoVolume("pnod2_FD3A", pvol, pmed);
      mod->AddNode(pnod2, 1, new TGeoTranslation(0, 0, -dpz));
    }
  }
  return mod;
}

TGeoVolumeAssembly* Detector::buildModuleC()
{
  TGeoVolumeAssembly* mod = new TGeoVolumeAssembly("FD3C");

  const TGeoMedium* medium = gGeoManager->GetMedium("FD3_Scintillator");

  float dphiDeg = 360. / mNumberOfSectors;

  for (int ir = 0; ir < mNumberOfRingsC; ir++) {
    std::string rName = "fd3_ring" + std::to_string(ir + 1 + mNumberOfRingsA);
    TGeoVolumeAssembly* ring = new TGeoVolumeAssembly(rName.c_str());
    float rmin = mRingSizesC[ir];
    float rmax = mRingSizesC[ir + 1];
    LOG(info) << "ring" << ir + mNumberOfRingsA << ": from " << rmin << " to " << rmax;
    for (int ic = 0; ic < mNumberOfSectors; ic++) {
      int cellId = ic + mNumberOfSectors * (ir + mNumberOfRingsA);
      std::string nodeName = "fd3_node" + std::to_string(cellId);
      float phimin = dphiDeg * ic;
      float phimax = dphiDeg * (ic + 1);
      auto tbs = new TGeoTubeSeg("tbs", rmin, rmax, mDzScint, phimin, phimax);
      auto nod = new TGeoVolume(nodeName.c_str(), tbs, medium);
      if ((ir + ic) % 2 == 0) {
        nod->SetLineColor(kBlue);
      } else {
        nod->SetLineColor(kBlue - 7);
      }
      ring->AddNode(nod, cellId);
    }
    mod->AddNode(ring, ir + 1);
  }

  // Aluminium plates on both sides of the C side module
  if (mFullContainer) {
    LOG(info) << "adding container on C side";
    auto pmed = (TGeoMedium*)gGeoManager->GetMedium("FD3_Aluminium");
    auto pvol = new TGeoTube("pvol_fd3c", mRingSizesC[0], mRingSizesC[mNumberOfRingsC], mDzPlate);
    auto pnod1 = new TGeoVolume("pnod1_FD3C", pvol, pmed);
    auto pnod2 = new TGeoVolume("pnod2_FD3C", pvol, pmed);
    double dpz = mDzScint / 2 + mDzPlate / 2;

    mod->AddNode(pnod1, 1, new TGeoTranslation(0, 0, dpz));
    mod->AddNode(pnod2, 2, new TGeoTranslation(0, 0, -dpz));
  }

  return mod;
}

void Detector::defineSensitiveVolumes()
{
  LOG(info) << "Adding FD3 Sentitive Volumes";

  TGeoVolume* v;
  TString volumeName;

  int nCellsA = mNumberOfRingsA * mNumberOfSectors;
  int nCellsC = mNumberOfRingsC * mNumberOfSectors;

  LOG(info) << "number of A rings = " << mNumberOfRingsA << " number of cells = " << nCellsA;
  LOG(info) << "number of C rings = " << mNumberOfRingsC << " number of cells = " << nCellsC;

  for (int iv = 0; iv < nCellsA + nCellsC; iv++) {
    volumeName = "fd3_node" + std::to_string(iv);
    v = gGeoManager->GetVolume(volumeName);
    LOG(info) << "Adding sensitive volume => " << v->GetName();
    AddSensitiveVolume(v);
  }
}

float Detector::ringSize(float z, float eta)
{
  return z * TMath::Tan(2 * TMath::ATan(TMath::Exp(-eta)));
}

ClassImp(o2::fd3::Detector);
