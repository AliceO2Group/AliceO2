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

#include "FT3Simulation/Detector.h"

#include "DetectorsBase/Stack.h"
#include "ITSMFTSimulation/Hit.h"
#include "SimulationDataFormat/TrackReference.h"

#include "FT3Base/FT3BaseParam.h"
#include "FT3Base/GeometryTGeo.h"
#include "FT3Simulation/FT3Layer.h"

// FairRoot includes
#include "FairDetector.h"    // for FairDetector
#include "FairRootManager.h" // for FairRootManager
#include "FairRootManager.h"
#include "FairRun.h"       // for FairRun
#include "FairRuntimeDb.h" // for FairRuntimeDb
#include "FairVolume.h"    // for FairVolume

#include "TGeoManager.h"     // for TGeoManager, gGeoManager
#include "TGeoPcon.h"        // for TGeoPcon
#include "TGeoTube.h"        // for TGeoTube
#include "TGeoVolume.h"      // for TGeoVolume, TGeoVolumeAssembly
#include "TString.h"         // for TString, operator+
#include "TVirtualMC.h"      // for gMC, TVirtualMC
#include "TVirtualMCStack.h" // for TVirtualMCStack

#include <fairlogger/Logger.h> // for LOG, LOG_IF

#include <cstdio> // for NULL, snprintf

#define MAX_SENSORS 2000

class FairModule;

class TGeoMedium;

class TParticle;

using namespace o2::ft3;
using o2::itsmft::Hit;

//_________________________________________________________________________________________________
Detector::Detector()
  : o2::base::DetImpl<Detector>("FT3", kTRUE),
    mTrackData(),
    mHits(o2::utils::createSimVector<o2::itsmft::Hit>())
{
}

//_________________________________________________________________________________________________
void Detector::buildBasicFT3(const FT3BaseParam& param)
{
  // Build a basic parametrized FT3 detector with nLayers equally spaced between z_first and z_first+z_length
  // Covering pseudo rapidity [etaIn,etaOut]. Silicon thinkness computed to match layer x/X0

  LOG(info) << "Building FT3 Detector: Conical Telescope";

  const int numberOfLayers = param.nLayers;
  const auto z_first = param.z0;
  const auto z_length = param.zLength;
  const auto etaIn = param.etaIn;
  const auto etaOut = param.etaOut;
  const auto Layerx2X0 = param.Layerx2X0;
  mLayerName[IdxBackwardDisks].resize(numberOfLayers);
  mLayerName[IdxForwardDisks].resize(numberOfLayers);

  for (int direction : {IdxBackwardDisks, IdxForwardDisks}) {
    for (int layerNumber = 0; layerNumber < numberOfLayers; layerNumber++) {
      std::string layerName = GeometryTGeo::getFT3LayerPattern() + std::to_string(layerNumber + numberOfLayers * direction);
      mLayerName[direction][layerNumber] = layerName;

      // Adds evenly spaced layers
      const float layerZ = z_first + (layerNumber * z_length / numberOfLayers) * std::copysign(1, z_first);
      const float rIn = std::abs(layerZ * std::tan(2.f * std::atan(std::exp(-etaIn))));
      const float rOut = std::abs(layerZ * std::tan(2.f * std::atan(std::exp(-etaOut))));
      const bool isMiddleLayer = layerNumber < 3;
      auto& thisLayer = mLayers[direction].emplace_back(direction, layerNumber, layerName, layerZ, rIn, rOut, Layerx2X0, isMiddleLayer);
    }
  }
}

//_________________________________________________________________________________________________
void Detector::buildFT3V1()
{
  // Build FT3 detector according to
  // https://indico.cern.ch/event/992488/contributions/4174473/attachments/2168881/3661331/tracker_parameters_werner_jan_11_2021.pdf

  LOG(info) << "Building FT3 Detector: V1";

  const int numberOfLayers = 10;
  const float sensorThickness = 30.e-4;
  const float layersx2X0 = 1.e-2;
  const std::vector<std::array<float, 4>> layersConfig{
    {26., .5, 3., 0.1f * layersx2X0}, // {z_layer, r_in, r_out, Layerx2X0}
    {30., .5, 3., 0.1f * layersx2X0},
    {34., .5, 3., 0.1f * layersx2X0},
    {77., 3.5, 35., layersx2X0},
    {100., 3.5, 35., layersx2X0},
    {122., 3.5, 35., layersx2X0},
    {150., 3.5, 80.f, layersx2X0},
    {180., 3.5, 80.f, layersx2X0},
    {220., 3.5, 80.f, layersx2X0},
    {279., 3.5, 80.f, layersx2X0}};

  mLayerName[IdxBackwardDisks].resize(numberOfLayers);
  mLayerName[IdxForwardDisks].resize(numberOfLayers);

  for (auto direction : {IdxBackwardDisks, IdxForwardDisks}) {
    for (int layerNumber = 0; layerNumber < numberOfLayers; layerNumber++) {
      std::string directionName = std::to_string(direction);
      std::string layerName = GeometryTGeo::getFT3LayerPattern() + directionName + std::string("_") + std::to_string(layerNumber);
      mLayerName[direction][layerNumber] = layerName;
      auto& z = layersConfig[layerNumber][0];

      auto& rIn = layersConfig[layerNumber][1];
      auto& rOut = layersConfig[layerNumber][2];
      auto& x0 = layersConfig[layerNumber][3];

      LOG(info) << "Adding Layer " << layerName << " at z = " << z;
      // Add layers
      const bool isMiddleLayer = layerNumber < 3;
      auto& thisLayer = mLayers[direction].emplace_back(direction, layerNumber, layerName, z, rIn, rOut, x0, isMiddleLayer);
    }
  }
}

//_________________________________________________________________________________________________
void Detector::buildFT3V3b()
{
  // Build FT3 detector according to
  // https://www.overleaf.com/project/6051acc870e39aaeb4653621

  LOG(info) << "Building FT3 Detector: V3b";

  const int numberOfLayers = 12;
  float sensorThickness = 30.e-4;
  float layersx2X0 = 1.e-2;
  std::vector<std::array<float, 4>> layersConfig{
    {26., .5, 3., 0.1f * layersx2X0}, // {z_layer, r_in, r_out, Layerx2X0}
    {30., .5, 3., 0.1f * layersx2X0},
    {34., .5, 3., 0.1f * layersx2X0},
    {77., 5.0, 35., layersx2X0},
    {100., 5.0, 35., layersx2X0},
    {122., 5.0, 35., layersx2X0},
    {150., 5.5, 80.f, layersx2X0},
    {180., 6.6, 80.f, layersx2X0},
    {220., 8.1, 80.f, layersx2X0},
    {279., 10.2, 80.f, layersx2X0},
    {340., 12.5, 80.f, layersx2X0},
    {400., 14.7, 80.f, layersx2X0}};

  mLayerName[IdxBackwardDisks].resize(numberOfLayers);
  mLayerName[IdxForwardDisks].resize(numberOfLayers);

  for (auto direction : {IdxBackwardDisks, IdxForwardDisks}) {
    for (int layerNumber = 0; layerNumber < numberOfLayers; layerNumber++) {
      std::string directionName = std::to_string(direction);
      std::string layerName = GeometryTGeo::getFT3LayerPattern() + directionName + std::string("_") + std::to_string(layerNumber);
      mLayerName[direction][layerNumber] = layerName;
      auto& z = layersConfig[layerNumber][0];

      auto& rIn = layersConfig[layerNumber][1];
      auto& rOut = layersConfig[layerNumber][2];
      auto& x0 = layersConfig[layerNumber][3];

      LOG(info) << "Adding Layer " << layerName << " at z = " << z;
      // Add layers
      const bool isMiddleLayer = layerNumber < 3;
      auto& thisLayer = mLayers[direction].emplace_back(direction, layerNumber, layerName, z, rIn, rOut, x0, isMiddleLayer);
    }
  }
}

void Detector::buildFT3NewVacuumVessel()
{
  // Build the FT3 detector according to changes proposed during
  // https://indico.cern.ch/event/1407704/
  // to adhere to the changes that were presented at the ALICE 3 Upgrade days in March 2024
  // Inner radius at C-side to 7 cm
  // Inner radius at A-side stays at 5 cm
  // 06.02.2025 update: IRIS layers are now in TRK

  LOG(info) << "Building FT3 Detector: After Upgrade Days March 2024 version";

  const int numberOfLayers = 9;
  const float sensorThickness = 30.e-4;
  const float layersx2X0 = 1.e-2;
  const std::vector<std::array<float, 4>> layersConfigCSide{
    {77., 7.0, 35., layersx2X0}, // {z_layer, r_in, r_out, Layerx2X0}
    {100., 7.0, 35., layersx2X0},
    {122., 7.0, 35., layersx2X0},
    {150., 7.0, 68.f, layersx2X0},
    {180., 7.0, 68.f, layersx2X0},
    {220., 7.0, 68.f, layersx2X0},
    {260., 7.0, 68.f, layersx2X0},
    {300., 7.0, 68.f, layersx2X0},
    {350., 7.0, 68.f, layersx2X0}};

  const std::vector<std::array<float, 4>> layersConfigASide{
    {77., 5.0, 35., layersx2X0}, // {z_layer, r_in, r_out, Layerx2X0}
    {100., 5.0, 35., layersx2X0},
    {122., 5.0, 35., layersx2X0},
    {150., 5.0, 68.f, layersx2X0},
    {180., 5.0, 68.f, layersx2X0},
    {220., 5.0, 68.f, layersx2X0},
    {260., 5.0, 68.f, layersx2X0},
    {300., 5.0, 68.f, layersx2X0},
    {350., 5.0, 68.f, layersx2X0}};

  mLayerName[IdxBackwardDisks].resize(numberOfLayers);
  mLayerName[IdxForwardDisks].resize(numberOfLayers);

  for (auto direction : {IdxBackwardDisks, IdxForwardDisks}) {
    for (int layerNumber = 0; layerNumber < numberOfLayers; layerNumber++) {
      std::string directionName = std::to_string(direction);
      std::string layerName = GeometryTGeo::getFT3LayerPattern() + directionName + std::string("_") + std::to_string(layerNumber);
      mLayerName[direction][layerNumber] = layerName;
      float z, rIn, rOut, x0;
      if (direction == 0) { // C-Side
        z = layersConfigCSide[layerNumber][0];
        rIn = layersConfigCSide[layerNumber][1];
        rOut = layersConfigCSide[layerNumber][2];
        x0 = layersConfigCSide[layerNumber][3];
      } else if (direction == 1) { // A-Side
        z = layersConfigASide[layerNumber][0];
        rIn = layersConfigASide[layerNumber][1];
        rOut = layersConfigASide[layerNumber][2];
        x0 = layersConfigASide[layerNumber][3];
      }

      LOG(info) << "Adding Layer " << layerName << " at z = " << z;
      // Add layers
      const bool isMiddleLayer = layerNumber < 3;
      auto& thisLayer = mLayers[direction].emplace_back(direction, layerNumber, layerName, z, rIn, rOut, x0, isMiddleLayer);
    }
  }
}

void Detector::buildFT3ScopingV3()
{
  // Build the FT3 detector according to v3 layout
  // https://indico.cern.ch/event/1596309/contributions/6728167/attachments/3190117/5677220/2025-12-10-AW-ALICE3planning.pdf
  // Middle disks inner radius 10 cm
  // Outer  disks inner radius 20 cm

  LOG(info) << "Building FT3 Detector: v3 scoping version";

  const int numberOfLayers = 6;
  const float sensorThickness = 30.e-4;
  const float layersx2X0 = 1.e-2;
  using LayerConfig = std::array<float, 4>; // {z_layer, r_in, r_out, Layerx2X0}
  const std::array<LayerConfig, numberOfLayers> layersConfigCSide{LayerConfig{77., 10.0, 35., layersx2X0},
                                                                  LayerConfig{100., 10.0, 35., layersx2X0},
                                                                  LayerConfig{122., 10.0, 35., layersx2X0},
                                                                  LayerConfig{150., 20.0, 68.f, layersx2X0},
                                                                  LayerConfig{180., 20.0, 68.f, layersx2X0},
                                                                  LayerConfig{220., 20.0, 68.f, layersx2X0}};

  const std::array<LayerConfig, numberOfLayers> layersConfigASide{LayerConfig{77., 10.0, 35., layersx2X0},
                                                                  LayerConfig{100., 10.0, 35., layersx2X0},
                                                                  LayerConfig{122., 10.0, 35., layersx2X0},
                                                                  LayerConfig{150., 20.0, 68.f, layersx2X0},
                                                                  LayerConfig{180., 20.0, 68.f, layersx2X0},
                                                                  LayerConfig{220., 20.0, 68.f, layersx2X0}};
  const std::array<bool, numberOfLayers> enabled{true, true, true, true, true, true}; // To enable or disable layers for debug purpose

  for (int direction : {IdxBackwardDisks, IdxForwardDisks}) {
    mLayerName[direction].clear();
    const std::array<LayerConfig, numberOfLayers>& layerConfig = (direction == IdxBackwardDisks) ? layersConfigCSide : layersConfigASide;
    for (int layerNumber = 0; layerNumber < numberOfLayers; layerNumber++) {
      if (!enabled[layerNumber]) {
        continue;
      }
      const std::string directionName = std::to_string(direction);
      const std::string layerName = GeometryTGeo::getFT3LayerPattern() + directionName + std::string("_") + std::to_string(layerNumber);
      mLayerName[direction].push_back(layerName.c_str());
      const float z = layerConfig[layerNumber][0];
      const float rIn = layerConfig[layerNumber][1];
      const float rOut = layerConfig[layerNumber][2];
      const float x0 = layerConfig[layerNumber][3];
      LOG(info) << "buildFT3ScopingV3 -> Adding Layer " << layerNumber << "/" << numberOfLayers << " " << layerName << " at z = " << z;
      // Add layers
      const bool isMiddleLayer = layerNumber < 3;
      auto& thisLayer = mLayers[direction].emplace_back(direction, layerNumber, layerName, z, rIn, rOut, x0, isMiddleLayer);
    }
  }
}

//_________________________________________________________________________________________________
void Detector::buildFT3Scoping()
{
  // Build FT3 detector according to the scoping document

  LOG(info) << "Building FT3 Detector: Scoping document version";

  const int numberOfLayers = 12;
  const float sensorThickness = 30.e-4;
  const float layersx2X0 = 1.e-2;
  const std::vector<std::array<float, 4>> layersConfig{
    {26., .5, 2.5, 0.1f * layersx2X0}, // {z_layer, r_in, r_out, Layerx2X0}
    {30., .5, 2.5, 0.1f * layersx2X0},
    {34., .5, 2.5, 0.1f * layersx2X0},
    {77., 5.0, 35., layersx2X0},
    {100., 5.0, 35., layersx2X0},
    {122., 5.0, 35., layersx2X0},
    {150., 5.0, 68.f, layersx2X0},
    {180., 5.0, 68.f, layersx2X0},
    {220., 5.0, 68.f, layersx2X0},
    {260., 5.0, 68.f, layersx2X0},
    {300., 5.0, 68.f, layersx2X0},
    {350., 5.0, 68.f, layersx2X0}};

  mLayerName[IdxBackwardDisks].resize(numberOfLayers);
  mLayerName[IdxForwardDisks].resize(numberOfLayers);

  for (auto direction : {IdxBackwardDisks, IdxForwardDisks}) {
    for (int layerNumber = 0; layerNumber < numberOfLayers; layerNumber++) {
      std::string directionName = std::to_string(direction);
      std::string layerName = GeometryTGeo::getFT3LayerPattern() + directionName + std::string("_") + std::to_string(layerNumber);
      mLayerName[direction][layerNumber] = layerName;
      auto& z = layersConfig[layerNumber][0];
      auto& rIn = layersConfig[layerNumber][1];
      auto& rOut = layersConfig[layerNumber][2];
      auto& x0 = layersConfig[layerNumber][3];

      LOG(info) << "Adding Layer " << layerName << " at z = " << z;
      // Add layers
      const bool isMiddleLayer = layerNumber < 3;
      auto& thisLayer = mLayers[direction].emplace_back(direction, layerNumber, layerName, z, rIn, rOut, x0, isMiddleLayer);
    }
  }
}

//_________________________________________________________________________________________________
Detector::Detector(bool active)
  : o2::base::DetImpl<Detector>("FT3", active),
    mTrackData(),
    mHits(o2::utils::createSimVector<o2::itsmft::Hit>())
{
  buildFT3ScopingV3(); // v3 Dec 25
}

//_________________________________________________________________________________________________
Detector::Detector(const Detector& rhs)
  : o2::base::DetImpl<Detector>(rhs),
    mTrackData(),

    /// Container for data points
    mHits(o2::utils::createSimVector<o2::itsmft::Hit>())
{
  mLayerName = rhs.mLayerName;
  mActiveSensorMap = rhs.mActiveSensorMap;
}

//_________________________________________________________________________________________________
Detector::~Detector()
{

  if (mHits) {
    // delete mHits;
    o2::utils::freeSimVector(mHits);
  }
}

//_________________________________________________________________________________________________
Detector& Detector::operator=(const Detector& rhs)
{
  // The standard = operator
  // Inputs:
  //   Detector   &h the sourse of this copy
  // Outputs:
  //   none.
  // Return:
  //  A copy of the sourse hit h

  if (this == &rhs) {
    return *this;
  }

  // base class assignment
  base::Detector::operator=(rhs);

  mLayerName = rhs.mLayerName;
  mActiveSensorMap = rhs.mActiveSensorMap;
  mLayers = rhs.mLayers;
  mTrackData = rhs.mTrackData;

  /// Container for data points
  mHits = nullptr;

  return *this;
}

//_________________________________________________________________________________________________
void Detector::InitializeO2Detector()
{
  // Define the list of sensitive volumes
  LOG(info) << "Initialize FT3 O2Detector";

  defineSensitiveVolumes();
}

//_________________________________________________________________________________________________
bool Detector::ProcessHits(FairVolume* vol)
{
  // This method is called from the MC stepping
  if (!(fMC->TrackCharge())) {
    return kFALSE;
  }

  int volID = vol->getMCid();

  auto it = mActiveSensorMap.find(volID);
  if (it == mActiveSensorMap.end()) {
    return kFALSE; // Not a sensitive volume
  }

  int lay = it->second;

  auto stack = (o2::data::Stack*)fMC->GetStack();

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
    return kFALSE; // do noting
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
    int chipindex = lay;

    Hit* p = addHit(stack->GetCurrentTrackNumber(), chipindex, mTrackData.mPositionStart.Vect(), positionStop.Vect(),
                    mTrackData.mMomentumStart.Vect(), mTrackData.mMomentumStart.E(), positionStop.T(),
                    mTrackData.mEnergyLoss, mTrackData.mTrkStatusStart, status);
    // p->SetTotalEnergy(vmc->Etot());

    // RS: not sure this is needed
    // Increment number of Detector det points in TParticle
    stack->addHit(GetDetId());
  }

  return kTRUE;
}

//_________________________________________________________________________________________________
void Detector::createMaterials()
{
  int ifield = 2;
  float fieldm = 10.0;
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

//_________________________________________________________________________________________________
void Detector::EndOfEvent() { Reset(); }

//_________________________________________________________________________________________________
void Detector::Register()
{
  // This will create a branch in the output tree called Hit, setting the last
  // parameter to kFALSE means that this collection will not be written to the file,
  // it will exist only during the simulation

  if (FairRootManager::Instance()) {
    FairRootManager::Instance()->RegisterAny(addNameTo("Hit").data(), mHits, kTRUE);
  }
}

//_________________________________________________________________________________________________
void Detector::Reset()
{
  if (!o2::utils::ShmManager::Instance().isOperational()) {
    mHits->clear();
  }
}

//_________________________________________________________________________________________________
void Detector::ConstructGeometry()
{
  // Create detector materials
  createMaterials();

  // Construct the detector geometry
  createGeometry();
}

//_________________________________________________________________________________________________
void Detector::createGeometry()
{

  TGeoVolume* volFT3 = new TGeoVolumeAssembly(GeometryTGeo::getFT3VolPattern());
  TGeoVolume* volIFT3 = new TGeoVolumeAssembly(GeometryTGeo::getFT3InnerVolPattern());

  LOG(info) << "FT3: createGeometry volume name = " << GeometryTGeo::getFT3VolPattern();

  TGeoVolume* vALIC = gGeoManager->GetVolume("barrel");
  if (!vALIC) {
    LOG(fatal) << "Could not find the top volume";
  }

  TGeoVolume* A3IPvac = gGeoManager->GetVolume("OUT_PIPEVACUUM");
  if (!A3IPvac) {
    LOG(info) << "Running simulation with no beam pipe.";
  }

  // This will need to adapt to the new scheme
  if (!A3IPvac) {
    for (int direction : {IdxBackwardDisks, IdxForwardDisks}) { // Backward layers at mLayers[0]; Forward layers at mLayers[1]
      const std::string directionString = direction ? "Forward" : "Backward";
      LOG(info) << "  Creating FT3 without beampipe " << directionString << " layers:";
      for (int iLayer = 0; iLayer < mLayers[direction].size(); iLayer++) {
        mLayers[direction][iLayer].createLayer(volFT3);
      }
    }
    vALIC->AddNode(volFT3, 2, new TGeoTranslation(0., 30., 0.));
  } else { // If beampipe is enabled append inner disks to beampipe filling volume, this should be temporary.
    for (int direction : {IdxBackwardDisks, IdxForwardDisks}) {
      const std::string directionString = direction ? "Forward" : "Backward";
      LOG(info) << "  Creating FT3 " << directionString << " layers:";
      for (int iLayer = 0; iLayer < mLayers[direction].size(); iLayer++) {
        LOG(info) << "  Creating " << directionString << " layer " << iLayer;
        if (mLayers[direction][iLayer].getIsInMiddleLayer()) { // ML disks
          mLayers[direction][iLayer].createLayer(volIFT3);
        } else {
          mLayers[direction][iLayer].createLayer(volFT3);
        }
      }
    }
    A3IPvac->AddNode(volIFT3, 2, new TGeoTranslation(0., 0., 0.));
    vALIC->AddNode(volFT3, 2, new TGeoTranslation(0., 30., 0.));
  }
}

//_________________________________________________________________________________________________
void Detector::defineSensitiveVolumes()
{
  TGeoManager* geoManager = gGeoManager;

  // Get the flat list of ALL volumes present in the geometry
  TObjArray* allVolumes = geoManager->GetListOfVolumes();
  int nVolumes = allVolumes->GetEntriesFast();

  LOG(info) << "Adding FT3 Sensitive Volumes by iterating over all geometry volumes...";

  for (int direction : {IdxBackwardDisks, IdxForwardDisks}) {
    for (int iLayer = 0; iLayer < getNumberOfLayers(); iLayer++) {
      int iSens = 0;

      // Build the "signatures" (prefixes) of the names for the various layouts for this specific layer and direction:

      // 1. Trapezoidal/Cylindrical (format: FT3Sensor_<dir>_<layer>)
      std::string sig1 = Form("%s_%d_%d", GeometryTGeo::getFT3SensorPattern(), direction, iLayer);

      // 2. Segmented front/back (format: FT3Sensor_front_<layer>_<dir>_...)
      std::string sig2 = "FT3Sensor_front_" + std::to_string(iLayer) + "_" + std::to_string(direction);
      std::string sig3 = "FT3Sensor_back_" + std::to_string(iLayer) + "_" + std::to_string(direction);

      // 3. SegmentedStave (format: FT3Sensor_<dir>_<layer>_...)
      // Add the trailing underscore to avoid confusing it with sig1
      std::string sig4 = "FT3Sensor_" + std::to_string(direction) + "_" + std::to_string(iLayer) + "_";

      // Iterate over all existing volumes to find matches
      for (int i = 0; i < nVolumes; ++i) {
        TGeoVolume* v = (TGeoVolume*)allVolumes->At(i);
        std::string vName = v->GetName();

        // Explicitly exclude the inactive silicon regions created in FT3Module
        if (vName.find("Inactive") != std::string::npos || vName.find("inactive") != std::string::npos) {
          continue;
        }

        // Check if the volume name matches one of our active sensors
        bool isMatch = false;
        if (vName == sig1) {
          isMatch = true; // Exact match for Trapezoidal/Cylindrical layouts
        } else if (vName.find(sig2) == 0 || vName.find(sig3) == 0 || vName.find(sig4) == 0) {
          isMatch = true; // Prefix match for Segmented and SegmentedStave layouts
        }

        if (isMatch) {
          AddSensitiveVolume(v);
          int volID = gMC ? TVirtualMC::GetMC()->VolId(vName.c_str()) : 0;
          if (volID > 0) {
            mActiveSensorMap[volID] = iLayer;
          }
          iSens++;
        }
      }

      if (iSens == 0) {
        LOG(error) << "NO sensitive volume found for direction " << direction << ", layer " << iLayer;
      } else {
        LOG(info) << iSens << " sensitive volume(s) added for direction " << direction << " layer " << iLayer;
      }
    }
  }
}

//_________________________________________________________________________________________________
Hit* Detector::addHit(int trackID, int detID, const TVector3& startPos, const TVector3& endPos,
                      const TVector3& startMom, double startE, double endTime, double eLoss, unsigned char startStatus,
                      unsigned char endStatus)
{
  mHits->emplace_back(trackID, detID, startPos, endPos, startMom, startE, endTime, eLoss, startStatus, endStatus);
  return &(mHits->back());
}

ClassImp(o2::ft3::Detector);
