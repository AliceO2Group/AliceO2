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
#include "ECalSimulation/Detector.h"
#include "ECalBase/ECalBaseParam.h"

using o2::itsmft::Hit;

namespace o2
{
namespace ecal
{

Detector::Detector()
  : o2::base::DetImpl<Detector>("ECL", true),
    mTrackData(),
    mHits(o2::utils::createSimVector<o2::itsmft::Hit>())
{
}

Detector::Detector(bool active)
  : o2::base::DetImpl<Detector>("ECL", true),
    mTrackData(),
    mHits(o2::utils::createSimVector<o2::itsmft::Hit>())
{
  auto& ecalPars = ECalBaseParam::Instance();
  mInnerRadius = ecalPars.rMin;
  mOuterRadius = ecalPars.rMax;
  mLength = ecalPars.zLength;
  mEnableEndcap = ecalPars.enableFwdEndcap;
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

  float tmaxfdLead = 0.1;        // .10000E+01; // Degree
  float stemaxLead = .10000E+01; // cm
  float deemaxLead = 0.1;        // 0.30000E-02; // Fraction of particle's energy 0<deemax<=1
  float epsilLead = 1.0E-4;      // .10000E+01;
  float stminLead = 0.0;         // cm "Default value used"

  // First approximation is a detector full of lead
  // Lead
  Detector::Material(1, "LEAD", 207.19, 82., 11.35, .56, 18.5);
  Detector::Medium(1, "LEAD", 1, 0, ifield, fieldm, tmaxfdLead, stemaxLead, deemaxLead, epsilLead, stminLead);
}

void Detector::InitializeO2Detector()
{
  LOG(info) << "Initialize ECal O2Detector";
  mGeometryTGeo = GeometryTGeo::Instance();
  // defineSensitiveVolumes();
}

void Detector::defineSensitiveVolumes()
{
  LOGP(info, "Adding ECal Sensitive Volumes");
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
  LOGP(info, "Creating ECal geometry");

  TGeoManager* geoManager = gGeoManager;
  TGeoVolume* vALIC = geoManager->GetVolume("barrel");
  if (!vALIC) {
    LOGP(fatal, "Could not find barrel volume while constructing ECal geometry");
  }
  new TGeoVolumeAssembly(GeometryTGeo::getECalVolPattern());
  TGeoVolume* vECal = geoManager->GetVolume(GeometryTGeo::getECalVolPattern());
  vALIC->AddNode(vECal, 2, new TGeoTranslation(0, 30., 0));

  char vstrng[100] = "ECalVol";
  vECal->SetTitle(vstrng);

  // Build the ECal cylinder
  auto& matmgr = o2::base::MaterialManager::Instance();
  TGeoMedium* medPb = matmgr.getTGeoMedium("ECL_LEAD");
  TGeoTube* ecalShape = new TGeoTube("ECLsh", mInnerRadius, mOuterRadius, mLength);
  TGeoVolume* ecalVol = new TGeoVolume("ECL", ecalShape, medPb);
  ecalVol->SetLineColor(kAzure - 9);
  ecalVol->SetTransparency(0);
  vECal->AddNode(ecalVol, 1, nullptr);

  if (mEnableEndcap) {
    // Build the ecal endcap
    TGeoTube* ecalEndcapShape = new TGeoTube("ECLECsh", 15.f, 160.f, 0.5 * (mOuterRadius - mInnerRadius));
    TGeoVolume* ecalEndcapVol = new TGeoVolume("ECLEC", ecalEndcapShape, medPb);
    ecalEndcapVol->SetLineColor(kAzure - 9);
    ecalEndcapVol->SetTransparency(0);
    vECal->AddNode(ecalEndcapVol, 1, new TGeoTranslation(0, 0, -450.f));
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

} // namespace ecal
} // namespace o2

ClassImp(o2::ecal::Detector);
O2DetectorCreatorImpl(o2::ecal::Detector::create, ecl);