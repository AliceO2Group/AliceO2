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

using o2::itsmft::Hit;

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
  configLayers(iotofPars.enableInnerTOF, iotofPars.enableOuterTOF, iotofPars.enableForwardTOF);
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

void Detector::configLayers(bool itof, bool otof, bool ftof, bool btof)
{
  if (itof) {
    mITOFLayer = ITOFLayer(std::string{GeometryTGeo::getITOFLayerPattern()}, 19.f, 0.f, 124.f, 0.f, 0.02f, true); // iTOF
  }
  if (otof) {
    mOTOFLayer = OTOFLayer(std::string{GeometryTGeo::getOTOFLayerPattern()}, 85.f, 0.f, 680.f, 0.f, 0.02f, true); // oTOF
  }
  if (ftof) {
    mFTOFLayer = FTOFLayer(std::string{GeometryTGeo::getFTOFLayerPattern()}, 15.f, 150.f, 0.f, 405.f, 0.02f, false); // fTOF
  }
  if (btof) {
    mBTOFLayer = BTOFLayer(std::string{GeometryTGeo::getBTOFLayerPattern()}, 15.f, 150.f, 0.f, -405.f, 0.02f, false); // bTOF
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
    v = geoManager->GetVolume(GeometryTGeo::getITOFSensorPattern());
    LOGP(info, "Adding IOTOF Sensitive Volume {}", v->GetName());
    AddSensitiveVolume(v);
  }
  if (iotofPars.enableOuterTOF) {
    v = geoManager->GetVolume(GeometryTGeo::getOTOFSensorPattern());
    LOGP(info, "Adding IOTOF Sensitive Volume {}", v->GetName());
    AddSensitiveVolume(v);
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

  // Is it needed to keep a track reference when the outer ITS volume is encountered?
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