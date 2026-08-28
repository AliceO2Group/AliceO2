// Copyright 2019-2026 CERN and copyright holders of ALICE O2.
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
  definePassiveVolumes();
}

bool Detector::ProcessHits(FairVolume* vol)
{
  // This method is called from the MC stepping
  if (!(fMC->TrackCharge())) {
    return kFALSE;
  }

  int copy = 0;
  int volId = fMC->CurrentVolID(copy);
  int detId = mChannelId[volId];

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

void Detector::ConstructGeometry()
{
  createMaterials();
  buildModules();
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

  int fieldType;
  float maxField;
  // TODO: Comment out two lines below once tested that the above function assigns field type and max correctly
  fieldType = 3;  // Field type
  maxField = 5.0; // Field max.
                  // o2::base::Detector::initFieldTrackingParams(fieldType, maxField);
  LOG(info) << "FD3: createMaterials(): fieldType " << fieldType << ", maxField " << maxField;

  float tmaxfd3 = -10.0;  // max deflection angle due to magnetic field in one step
  float stepmax = 0.1;    // max step allowed [cm]
  float deemax = 1.0;     // maximum fractional energy loss in one step 0<deemax<=1
  float epsil = 0.03;     // tracking precision [cm]
  float stepmin = -0.001; // minimum step due to continuous processes [cm] (negative value: choose it automatically)

  LOG(info) << "FD3: CreateMaterials(): fieldType " << fieldType << ", maxField " << maxField;

  o2::base::Detector::Mixture(++matId, "Scintillator", aScint, zScint, dScint, nScint, wScint);
  o2::base::Detector::Medium(Scintillator, "Scintillator", matId, sens, fieldType, maxField,
                             tmaxfd3, stepmax, deemax, epsil, stepmin);

  o2::base::Detector::Material(++matId, "Aluminium", aAlu, zAlu, dAlu, 8.9, 999);
  o2::base::Detector::Medium(Aluminium, "Aluminium", matId, unsens, fieldType, maxField,
                             tmaxfd3, stepmax, deemax, epsil, stepmin);

  // Cherenkov radiator
  fieldType = 2;  // magneticField->Integ();
  maxField = 10.; // magneticField->Max();

  // Radiator  glass SiO2
  Float_t aglass[2] = {28.0855, 15.9994};
  Float_t zglass[2] = {14., 8.};
  Float_t wglass[2] = {1., 2.};
  Float_t dglass = 2.2;

  // MCP glass SiO2
  Float_t dglass_mcp = 1.3;

  o2::base::Detector::Mixture(++matId, "MCP glass", aglass, zglass, dglass_mcp, -2, wglass);
  o2::base::Detector::Medium(MCPGlass, "Glass", matId, sens, fieldType, maxField,
                             10., .01, .1, .003, .003);
  o2::base::Detector::Mixture(++matId, "Radiator optical glass", aglass, zglass, dglass, -2, wglass);
  o2::base::Detector::Medium(RadiatorOpticalGlass, "OpticalGlass$", matId, sens, fieldType, maxField,
                             10., .01, .1, .003, .01);
}

void Detector::buildModules()
{
  LOGP(info, "Creating FD3 geometry");

  mNumberOfRingsScint = Constants::nringsScint;
  mNumberOfRingsCher = Constants::nringsCher;
  mNumberOfSectors = Constants::nsect;
  mDzScint = Constants::dzscint;
  mDzCher = Constants::dzcher;

  auto& baseParam = FD3BaseParam::Instance();

  mEtaMinScintA = Constants::etaMin_scintA;
  mEtaMinCherA = Constants::etaMin_cherA;
  mEtaMaxScintA = baseParam.isSymmetric ? Constants::etaMax_scintA_v1 : Constants::etaMax_scintA_v2;
  mEtaMaxCherA = baseParam.isSymmetric ? Constants::etaMax_cherA_v1 : Constants::etaMax_cherA_v2;

  mEtaMinScintC = Constants::etaMin_scintC;
  mEtaMinCherC = Constants::etaMin_cherC;
  mEtaMaxScintC = Constants::etaMax_scintC;
  mEtaMaxCherC = Constants::etaMax_cherC;

  mZScint = Constants::zscint;
  mZCher = Constants::zcher;

  auto topVolume = (TGeoVolume*)gGeoManager->GetVolume("barrel");

  mChannelCounter = 0;

  TGeoVolumeAssembly* vFD3_ScintA = buildModuleScintA();
  TGeoVolumeAssembly* vFD3_ScintC = buildModuleScintC();
  TGeoVolumeAssembly *vFD3_CherA, *vFD3_CherC;

  if (baseParam.isCherenkovSegmented) {
    if (baseParam.isSymmetric) {
      vFD3_CherA = buildModuleCherenkov_v1();
    } else {
      vFD3_CherA = buildModuleCherenkov_v2();
    }
    vFD3_CherC = buildModuleCherenkov_v1();
  } else {
    vFD3_CherA = buildModuleCherenkov_v0(mEtaMaxCherA, mEtaMaxCherA, mZCher);
    vFD3_CherC = buildModuleCherenkov_v0(mEtaMinCherC, mEtaMaxCherC, -mZCher);
  }

  vFD3_CherA->SetName("FD3_CherA");
  vFD3_CherC->SetName("FD3_CherC");

  topVolume->AddNode(vFD3_ScintA, 1, new TGeoTranslation(0., 30.f, mZScint));
  topVolume->AddNode(vFD3_ScintC, 1, new TGeoTranslation(0., 30.f, -mZScint));

  topVolume->AddNode(vFD3_CherA, 1, new TGeoTranslation(0., 30.f, mZCher));
  topVolume->AddNode(vFD3_CherC, 1, new TGeoTranslation(0., 30.f, -mZCher));
}

TGeoVolumeAssembly* Detector::buildModuleScintA()
{
  auto mod = new TGeoVolumeAssembly("FD3_ScintA");

  const TGeoMedium* medium = gGeoManager->GetMedium("FD3_Scintillator");

  float dphiDeg = 360. / mNumberOfSectors;

  for (int ir = 0; ir < mNumberOfRingsScint; ir++) {
    std::string rName = "fd3_ring" + std::to_string(ir + 1);
    float etaMin = mEtaMaxScintA - (ir + 1) * (mEtaMaxScintA - mEtaMinScintA) / mNumberOfRingsScint;
    float etaMax = mEtaMaxScintA - ir * (mEtaMaxScintA - mEtaMinScintA) / mNumberOfRingsScint;
    float zmod = mZScint;
    float rmin = getRingSize(zmod, etaMax), rmax = getRingSize(zmod, etaMin);
    LOG(info) << "Scintillator ring" << ir << ": from " << rmin << " to " << rmax;
    for (int ic = 0; ic < mNumberOfSectors; ic++) {
      int cellId = mChannelCounter++; // ic + mNumberOfSectors * ir;
      std::string nodeName = "fd3_node" + std::to_string(cellId);
      float phimin = dphiDeg * ic;
      float phimax = dphiDeg * (ic + 1);
      auto tbs = new TGeoTubeSeg("tbs", rmin, rmax, mDzScint / 2, phimin, phimax);
      auto node = new TGeoVolume(nodeName.c_str(), tbs, medium);
      node->SetLineColor((ir + ic) % 2 == 0 ? kRed : kRed - 7);
      mod->AddNode(node, 1);
    }
  }

  return mod;
}

TGeoVolumeAssembly* Detector::buildModuleScintC()
{
  auto mod = new TGeoVolumeAssembly("FD3_ScintC");

  const TGeoMedium* medium = gGeoManager->GetMedium("FD3_Scintillator");

  float dphiDeg = 360. / mNumberOfSectors;

  for (int ir = 0; ir < mNumberOfRingsScint; ir++) {
    std::string rName = "fd3_ring" + std::to_string(ir + 1 + mNumberOfRingsScint);
    float etaMin = mEtaMinScintC + ir * (mEtaMaxScintC - mEtaMinScintC) / mNumberOfRingsScint;
    float etaMax = mEtaMinScintC + (ir + 1) * (mEtaMaxScintC - mEtaMinScintC) / mNumberOfRingsScint;
    float zmod = -mZScint;
    float rmin = getRingSize(zmod, etaMin), rmax = getRingSize(zmod, etaMax);
    LOG(info) << "Scintillator ring" << ir + mNumberOfRingsScint << ": from " << rmin << " to " << rmax;
    for (int ic = 0; ic < mNumberOfSectors; ic++) {
      int cellId = mChannelCounter++; // ic + mNumberOfSectors * (ir + mNumberOfRingsScint);
      std::string nodeName = "fd3_node" + std::to_string(cellId);
      float phimin = dphiDeg * ic;
      float phimax = dphiDeg * (ic + 1);
      auto tbs = new TGeoTubeSeg("tbs", rmin, rmax, mDzScint / 2, phimin, phimax);
      auto node = new TGeoVolume(nodeName.c_str(), tbs, medium);
      node->SetLineColor((ir + ic) % 2 == 0 ? kBlue : kBlue - 7);
      mod->AddNode(node, 1);
    }
  }

  return mod;
}

TGeoVolumeAssembly* Detector::buildModuleCherenkov_v0(float etaMin, float etaMax, float zmod)
{
  auto mod = new TGeoVolumeAssembly("");

  const TGeoMedium* medium = gGeoManager->GetMedium("FD3_Glass");

  float rmin, rmax;
  if (zmod >= 0) {
    rmin = getRingSize(zmod, etaMin);
    rmax = getRingSize(zmod, etaMax);
  } else {
    rmin = getRingSize(zmod, etaMin);
    rmax = getRingSize(zmod, etaMax);
  }

  int cellId = mChannelCounter++;
  std::string nodeName = "fd3_node" + std::to_string(cellId);

  auto tbs = new TGeoTubeSeg("tbs", rmin, rmax, mDzScint / 2, 0, 360);
  auto node = new TGeoVolume(nodeName.c_str(), tbs, medium);
  node->SetLineColor(zmod > 0 ? kOrange : kMagenta);
  mod->AddNode(node, 1);

  return mod;
}

TGeoVolumeAssembly* Detector::buildModuleCherenkov_v1()
{
  auto mod = new TGeoVolumeAssembly("FD3_Ch");

  TGeoMedium* medium = gGeoManager->GetMedium("FD3_Glass");

  double rsize = Constants::rsize;

  const int N = 68;

  double x[N] = {7.5, 7.5, 6.4, 3.2, 0, -3.2, -6.4, -7.5, -7.5, -7.5, -6.4, -3.2,
                 0, 3.2, 6.4, 7.5, 10.7, 10.7, 9.6, 9.6, 6.4, 3.2, 0, -3.2,
                 -6.4, -9.6, -9.6, -10.7, -10.7, -10.7, -9.6, -9.6, -6.4, -3.2, 0, 3.2,
                 6.4, 9.6, 9.6, 10.7, 13.9, 13.9, 12.8, 12.8, 9.6, 6.4, 3.2, 0,
                 -3.2, -6.4, -9.6, -12.8, -12.8, -13.9, -13.9, -13.9, -12.8, -12.8, -9.6, -6.4,
                 -3.2, 0, 3.2, 6.4, 9.6, 12.8, 12.8, 13.9};

  double y[N] = {0, 3.2, 6.4, 7.5, 7.5, 7.5, 6.4, 3.2, 0, -3.2, -6.4, -7.5,
                 -7.5, -7.5, -6.4, -3.2, 0, 3.2, 6.4, 9.6, 9.6, 10.7, 10.7, 10.7,
                 9.6, 9.6, 6.4, 3.2, 0, -3.2, -6.4, -9.6, -9.6, -10.7, -10.7, -10.7,
                 -9.6, -9.6, -6.4, -3.2, 0, 3.2, 6.4, 9.6, 12.8, 12.8, 13.9, 13.9,
                 13.9, 12.8, 12.8, 9.6, 6.4, 3.2, 0, -3.2, -6.4, -9.6, -12.8, -12.8,
                 -13.9, -13.9, -13.9, -12.8, -12.8, -9.6, -6.4, -3.2};

  for (int i = 0; i < N; i++) {
    int cellId = mChannelCounter++;
    std::string nodeName = "fd3_node" + std::to_string(cellId);
    auto box = new TGeoBBox(rsize - 0.05, rsize - 0.05, mDzCher / 2);
    auto node = new TGeoVolume(nodeName.c_str(), box, medium);
    node->SetLineColor(kOrange + 7);
    mod->AddNode(node, 1, new TGeoTranslation(x[i], y[i], 0));
  }

  return mod;
}

TGeoVolumeAssembly* Detector::buildModuleCherenkov_v2()
{
  auto mod = new TGeoVolumeAssembly("FD3_Ch");

  TGeoMedium* medium = gGeoManager->GetMedium("FD3_Glass");

  double rsize = Constants::rsize;

  const int N = 68;

  double x[N] = {4.6, 3.2, 0, -3.2, -4.6, -3.2, 0, 3.2, 7.8, 6.4, 6.4, 3.2,
                 0, -3.2, -6.4, -6.4, -7.8, -6.4, -6.4, -3.2, 0, 3.2, 6.4, 6.4,
                 11, 9.6, 9.6, 9.6, 6.4, 3.2, 0, -3.2, -6.4, -9.6, -9.6, -9.6,
                 -11, -9.6, -9.6, -9.6, -6.4, -3.2, 0, 3.2, 6.4, 9.6, 9.6, 9.6,
                 14.2, 12.8, 12.8, 6.4, 3.2, 0, -3.2, -6.4, -12.8, -12.8, -14.2, -12.8,
                 -12.8, -6.4, -3.2, 0, 3.2, 6.4, 12.8, 12.8};

  double y[N] = {0, 3.2, 4.6, 3.2, 0, -3.2, -4.6, -3.2, 0, 3.2, 6.4, 6.4,
                 7.8, 6.4, 6.4, 3.2, 0, -3.2, -6.4, -6.4, -7.8, -6.4, -6.4, -3.2,
                 0, 3.2, 6.4, 9.6, 9.6, 9.6, 11, 9.6, 9.6, 9.6, 6.4, 3.2,
                 0, -3.2, -6.4, -9.6, -9.6, -9.6, -11, -9.6, -9.6, -9.6, -6.4, -3.2,
                 0, 3.2, 6.4, 12.8, 12.8, 14.2, 12.8, 12.8, 6.4, 3.2, 0, -3.2,
                 -6.4, -12.8, -12.8, -14.2, -12.8, -12.8, -6.4, -3.2};

  for (int i = 0; i < N; i++) {
    int cellId = mChannelCounter++;
    std::string nodeName = "fd3_node" + std::to_string(cellId);
    auto box = new TGeoBBox(rsize - 0.05, rsize - 0.05, mDzCher / 2);
    auto node = new TGeoVolume(nodeName.c_str(), box, medium);
    node->SetLineColor(kOrange + 7);
    mod->AddNode(node, 1, new TGeoTranslation(x[i], y[i], 0));
  }

  return mod;
}

void Detector::defineSensitiveVolumes()
{
  LOG(info) << "Adding FD3 sentitive volumes...";

  mChannelId = {};

  for (int ivol = 0; ivol < mChannelCounter; ivol++) {
    std::string volumeName = "fd3_node" + std::to_string(ivol);
    auto v = (TGeoVolume*)gGeoManager->GetVolume(volumeName.c_str());
    if (!v)
      continue;
    int volId = registerSensitiveVolumeAndGetVolID(v);
    mChannelId[volId] = ivol;
  }
  LOG(info) << "Done";
}

void Detector::definePassiveVolumes()
{
  // To be added later
}

float Detector::getRingSize(float z, float eta)
{
  return z * TMath::Tan(2 * TMath::ATan(TMath::Exp(-eta)));
}

ClassImp(o2::fd3::Detector);
