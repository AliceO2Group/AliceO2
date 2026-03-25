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

#include "DetectorsBase/Stack.h"
#include "ITSMFTSimulation/Hit.h"
#include "IOTOFSimulation/Detector.h"
#include "IOTOFBase/IOTOFBaseParam.h"

namespace o2
{
namespace iotof
{

Detector::Detector()
  : o2::base::DetImpl<Detector>("TF3", true),
    mTrackData(),
    mHits(o2::utils::createSimVector<o2::itsmft::Hit>())
{
}

Detector::Detector(bool active)
  : o2::base::DetImpl<Detector>("TF3", true),
    mTrackData(),
    mHits(o2::utils::createSimVector<o2::itsmft::Hit>())
{
  auto& iotofPars = IOTOFBaseParam::Instance();
  configLayers(iotofPars.enableInnerTOF, iotofPars.enableOuterTOF,
               iotofPars.enableForwardTOF, iotofPars.enableBackwardTOF,
               iotofPars.detectorPattern,
               iotofPars.segmentedInnerTOF, iotofPars.segmentedOuterTOF, iotofPars.x2x0);
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

void Detector::configLayers(bool itof, bool otof, bool ftof, bool btof, std::string pattern, bool itofSegmented, bool otofSegmented,
                            const float x2x0)
{

  const std::pair<float, float> dInnerTof = {21.f, 129.f}; // Radius and length
  std::pair<float, float> dOuterTof = {92.f, 680.f};       // Radius and length
  std::pair<float, float> radiusRangeDiskTof = {15.f, 100.f};
  float zForwardTof = 370.f;
  LOG(info) << "Configuring IOTOF layers with '" << pattern << "' pattern";
  if (pattern == "") {
    LOG(info) << "Default pattern";
  } else if (pattern == "v3b") {
    ftof = false;
    btof = false;
  } else if (pattern == "v3b1a") {
    dOuterTof.second = 500.f;
    zForwardTof = 270.f;
    radiusRangeDiskTof = {30.f, 100.f};
  } else if (pattern == "v3b1b") {
    dOuterTof.second = 500.f;
    zForwardTof = 200.f;
    radiusRangeDiskTof = {20.f, 68.f};
  } else if (pattern == "v3b2a") {
    dOuterTof.second = 440.f;
    zForwardTof = 270.f;
    radiusRangeDiskTof = {30.f, 120.f};
  } else if (pattern == "v3b2b") {
    dOuterTof.second = 440.f;
    zForwardTof = 200.f;
    radiusRangeDiskTof = {20.f, 68.f};
  } else if (pattern == "v3b3") {
    dOuterTof.second = 580.f;
    zForwardTof = 200.f;
    radiusRangeDiskTof = {20.f, 68.f};
  } else {
    LOG(fatal) << "IOTOF layer pattern " << pattern << " not recognized, exiting";
  }
  if (itof) { // iTOF
    const std::string name = GeometryTGeo::getITOFLayerPattern();
    const int nStaves = itofSegmented ? 24 : 0;              // number of staves in segmented case
    const double staveWidth = itofSegmented ? 5.42 : 0.0;    // cm
    const double staveTiltAngle = itofSegmented ? 3.0 : 0.0; // degrees
    const int modulesPerStave = itofSegmented ? 10 : 0;      // number of modules per stave in segmented case
    mITOFLayer = ITOFLayer(name,
                           dInnerTof.first, 0.f, dInnerTof.second, 0.f, x2x0, itofSegmented ? ITOFLayer::kBarrelSegmented : ITOFLayer::kBarrel,
                           nStaves, staveWidth, staveTiltAngle, modulesPerStave);
  }
  if (otof) { // oTOF
    const std::string name = GeometryTGeo::getOTOFLayerPattern();
    const int nStaves = otofSegmented ? 62 : 0;              // number of staves in segmented case
    const double staveWidth = otofSegmented ? 9.74 : 0.0;    // cm
    const double staveTiltAngle = otofSegmented ? 5.0 : 0.0; // degrees
    const int modulesPerStave = otofSegmented ? 54 : 0;      // number of modules per stave in segmented case
    mOTOFLayer = OTOFLayer(name,
                           dOuterTof.first, 0.f, dOuterTof.second, 0.f, x2x0, otofSegmented ? OTOFLayer::kBarrelSegmented : OTOFLayer::kBarrel,
                           nStaves, staveWidth, staveTiltAngle, modulesPerStave);
  }
  if (ftof) {
    const std::string name = GeometryTGeo::getFTOFLayerPattern();
    mFTOFLayer = FTOFLayer(name, radiusRangeDiskTof.first, radiusRangeDiskTof.second, 0.f, zForwardTof, x2x0, FTOFLayer::kDisk); // fTOF
  }
  if (btof) {
    const std::string name = GeometryTGeo::getBTOFLayerPattern();
    mBTOFLayer = BTOFLayer(name, radiusRangeDiskTof.first, radiusRangeDiskTof.second, 0.f, -zForwardTof, x2x0, BTOFLayer::kDisk); // bTOF
  }
}

void Detector::configServices()
{
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

  // AIR
  float aAir[4] = {12.0107, 14.0067, 15.9994, 39.948};
  float zAir[4] = {6., 7., 8., 18.};
  float wAir[4] = {0.000124, 0.755267, 0.231781, 0.012827};
  float dAir = 1.20479E-3;

  o2::base::Detector::Mixture(1, "AIR$", aAir, zAir, dAir, 4, wAir);
  o2::base::Detector::Medium(1, "AIR$", 1, 0, ifield, fieldm, tmaxfdAir, stemaxAir, deemaxAir, epsilAir, stminAir);

  o2::base::Detector::Material(3, "SILICON$", 0.28086E+02, 0.14000E+02, 0.23300E+01, 0.93600E+01, 0.99900E+03);
  o2::base::Detector::Medium(3, "SILICON$", 3, 0, ifield, fieldm, tmaxfdSi, stemaxSi, deemaxSi, epsilSi, stminSi);
}

void Detector::createGeometry()
{
  TGeoManager* geoManager = gGeoManager;
  TGeoVolume* vALIC = geoManager->GetVolume("barrel");
  if (!vALIC) {
    LOGP(fatal, "Could not find barrel volume while constructing IOTOF geometry");
  }
  new TGeoVolumeAssembly(GeometryTGeo::getIOTOFVolPattern());
  TGeoVolume* vIOTOF = geoManager->GetVolume(GeometryTGeo::getIOTOFVolPattern());
  vALIC->AddNode(vIOTOF, 2, new TGeoTranslation(0, 30., 0));

  char vstrng[100] = "IOTOFVol";
  vIOTOF->SetTitle(vstrng);

  auto& iotofPars = IOTOFBaseParam::Instance();
  if (iotofPars.enableInnerTOF) {
    mITOFLayer.createLayer(vIOTOF);
  }
  if (iotofPars.enableOuterTOF) {
    mOTOFLayer.createLayer(vIOTOF);
  }
  if (iotofPars.enableForwardTOF) {
    mFTOFLayer.createLayer(vIOTOF);
  }
  if (iotofPars.enableBackwardTOF) {
    mBTOFLayer.createLayer(vIOTOF);
  }
}

void Detector::InitializeO2Detector()
{
  LOG(info) << "Initialize IOTOF O2Detector";
  mGeometryTGeo = GeometryTGeo::Instance();
  defineSensitiveVolumes();
}

void Detector::defineSensitiveVolumes()
{
  TGeoManager* geoManager = gGeoManager;
  TGeoVolume* v;

  // The names of the IOTOF sensitive volumes have the format: IOTOFLayer(0...mLayers.size()-1)
  auto& iotofPars = IOTOFBaseParam::Instance();
  if (iotofPars.enableInnerTOF) {
    for (const std::string& itofSensor : ITOFLayer::mRegister) {
      v = geoManager->GetVolume(itofSensor.c_str());
      LOGP(info, "Adding IOTOF Sensitive Volume {}", v->GetName());
      AddSensitiveVolume(v);
    }
  }
  if (iotofPars.enableOuterTOF) {
    for (const std::string& otofSensor : OTOFLayer::mRegister) {
      v = geoManager->GetVolume(otofSensor.c_str());
      LOGP(info, "Adding IOTOF Sensitive Volume {}", v->GetName());
      AddSensitiveVolume(v);
    }
  }
  if (iotofPars.enableForwardTOF) {
    v = geoManager->GetVolume(GeometryTGeo::getFTOFSensorPattern());
    LOGP(info, "Adding IOTOF Sensitive Volume {}", v->GetName());
    AddSensitiveVolume(v);
  }
  if (iotofPars.enableBackwardTOF) {
    v = geoManager->GetVolume(GeometryTGeo::getBTOFSensorPattern());
    LOGP(info, "Adding IOTOF Sensitive Volume {}", v->GetName());
    AddSensitiveVolume(v);
  }
}

void Detector::EndOfEvent() { Reset(); }

void Detector::Register()
{
  // This will create a branch in the output tree called Hit, setting the last
  // parameter to false means that this collection will not be written to the file,
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

  // Is it needed to keep a track reference when the outer volume is encountered?
  auto stack = (o2::data::Stack*)fMC->GetStack();
  if (fMC->IsTrackExiting() /*&& (lay == 0 || lay == mLayers.size() - 1)*/) {
    // Keep the track refs for the innermost and outermost layers only
    o2::TrackReference tr(*fMC, GetDetId());
    tr.setTrackID(stack->GetCurrentTrackNumber());
    tr.setUserId(lay);
    stack->addTrackReference(tr);
  }
  bool startHit = false, stopHit = false;
  unsigned char status = 0;
  if (fMC->IsTrackEntering()) {
    status |= o2::itsmft::Hit::kTrackEntering;
  }
  if (fMC->IsTrackInside()) {
    status |= o2::itsmft::Hit::kTrackInside;
  }
  if (fMC->IsTrackExiting()) {
    status |= o2::itsmft::Hit::kTrackExiting;
  }
  if (fMC->IsTrackOut()) {
    status |= o2::itsmft::Hit::kTrackOut;
  }
  if (fMC->IsTrackStop()) {
    status |= o2::itsmft::Hit::kTrackStopped;
  }
  if (fMC->IsTrackAlive()) {
    status |= o2::itsmft::Hit::kTrackAlive;
  }

  // track is entering or created in the volume
  if ((status & o2::itsmft::Hit::kTrackEntering) || (status & o2::itsmft::Hit::kTrackInside && !mTrackData.mHitStarted)) {
    startHit = true;
  } else if ((status & (o2::itsmft::Hit::kTrackExiting | o2::itsmft::Hit::kTrackOut | o2::itsmft::Hit::kTrackStopped))) {
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

    o2::itsmft::Hit* p = addHit(stack->GetCurrentTrackNumber(), lay, mTrackData.mPositionStart.Vect(), positionStop.Vect(),
                                mTrackData.mMomentumStart.Vect(), mTrackData.mMomentumStart.E(), positionStop.T(),
                                mTrackData.mEnergyLoss, mTrackData.mTrkStatusStart, status);

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
} // namespace iotof
} // namespace o2

ClassImp(o2::iotof::Detector);
