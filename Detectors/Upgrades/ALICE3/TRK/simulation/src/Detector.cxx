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
#include "TRKSimulation/Hit.h"
#include "TRKSimulation/Detector.h"
#include "TRKBase/TRKBaseParam.h"
#include "TRKSimulation/VDGeometryBuilder.h"
#include "TRKSimulation/VDSensorRegistry.h"

#include <string>
#include <type_traits>

using o2::trk::Hit;

namespace o2
{
namespace trk
{

float getDetLengthFromEta(const float eta, const float radius)
{
  return 2. * (10. + radius * std::cos(2 * std::atan(std::exp(-eta))));
}

Detector::Detector()
  : o2::base::DetImpl<Detector>("TRK", true),
    mTrackData(),
    mHits(o2::utils::createSimVector<o2::trk::Hit>())
{
}

Detector::Detector(bool active)
  : o2::base::DetImpl<Detector>("TRK", true),
    mTrackData(),
    mHits(o2::utils::createSimVector<o2::trk::Hit>())
{
  auto& trkPars = TRKBaseParam::Instance();

  if (trkPars.configFile != "") {
    configFromFile(trkPars.configFile);
  } else {
    configMLOT();
    configToFile();
    configServices();
  }

  LOGP(info, "Summary of TRK configuration:");
  for (auto& layer : mLayers) {
    LOGP(info, "Layer: {} name: {} r: {} cm | z: {} cm | thickness: {} cm", layer->getNumber(), layer->getName(), layer->getInnerRadius(), layer->getZ(), layer->getChipThickness());
  }
}

Detector::Detector(const Detector& other)
  : o2::base::DetImpl<Detector>(other),
    mTrackData(),
    mHits(o2::utils::createSimVector<o2::trk::Hit>())
{
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

void Detector::configMLOT()
{
  auto& trkPars = TRKBaseParam::Instance();

  mLayers.clear();

  const std::vector<float> rInn{7.f, 9.f, 12.f, 20.f, 30.f, 45.f, 60.f, 80.f};
  const float thick = 100.e-3;

  switch (trkPars.layoutMLOT) {
    case kCylindrical: {
      const std::vector<float> length{128.35f, 128.35f, 128.35f, 128.35f, 128.35f, 256.7f, 256.7f, 256.7f};
      LOGP(warning, "Loading cylindrical configuration for ALICE3 TRK");
      for (int i{0}; i < 8; ++i) {
        std::string name = GeometryTGeo::getTRKLayerPattern() + std::to_string(i);
        mLayers.push_back(std::make_unique<TRKCylindricalLayer>(i, name, rInn[i], length[i], thick, MatBudgetParamMode::Thickness));
      }
      break;
    }
    case kSegmented: {
      const std::vector<int> nMods{10, 10, 10, 10, 10, 20, 20, 20};
      LOGP(warning, "Loading segmented configuration for ALICE3 TRK");
      for (int i{0}; i < 8; ++i) {
        std::string name = GeometryTGeo::getTRKLayerPattern() + std::to_string(i);
        if (i < 4) {
          mLayers.push_back(std::make_unique<TRKMLLayer>(i, name, rInn[i], nMods[i], thick, MatBudgetParamMode::Thickness));
        } else {
          mLayers.push_back(std::make_unique<TRKOTLayer>(i, name, rInn[i], nMods[i], thick, MatBudgetParamMode::Thickness));
        }
      }
      break;
    }
    default:
      LOGP(fatal, "Unknown option {} for configMLOT", static_cast<int>(trkPars.layoutMLOT));
      break;
  }
}

void Detector::configFromFile(std::string fileName)
{
  // Override the default geometry if config file provided
  std::ifstream confFile(fileName);
  if (!confFile.good()) {
    LOGP(fatal, "File {} not found, aborting.", fileName);
  }

  auto& trkPars = TRKBaseParam::Instance();

  mLayers.clear();

  LOGP(info, "Overriding geometry of ALICE3 TRK using {} file.", fileName);

  std::string line;
  std::vector<float> tmpBuff;
  int layerCount{0};
  while (std::getline(confFile, line)) {
    if (line[0] == '/') {
      continue;
    }
    tmpBuff.clear();
    std::stringstream ss(line);
    float val;
    std::string substr;
    while (getline(ss, substr, '\t')) {
      tmpBuff.push_back(std::stof(substr));
    }

    std::string name = GeometryTGeo::getTRKLayerPattern() + std::to_string(layerCount);
    switch (trkPars.layoutMLOT) {
      case kCylindrical:
        mLayers.push_back(std::make_unique<TRKCylindricalLayer>(layerCount, name, tmpBuff[0], tmpBuff[1], tmpBuff[2], MatBudgetParamMode::Thickness));
        break;
      case kSegmented: {
        int nMods = static_cast<int>(tmpBuff[1]);
        if (layerCount < 4) {
          mLayers.push_back(std::make_unique<TRKMLLayer>(layerCount, name, tmpBuff[0], nMods, tmpBuff[2], MatBudgetParamMode::Thickness));
        } else {
          mLayers.push_back(std::make_unique<TRKOTLayer>(layerCount, name, tmpBuff[0], nMods, tmpBuff[2], MatBudgetParamMode::Thickness));
        }
        break;
      }
      default:
        LOGP(fatal, "Unknown option {} for configMLOT", static_cast<int>(trkPars.layoutMLOT));
        break;
    }

    ++layerCount;
  }
}

void Detector::configToFile(std::string fileName)
{
  LOGP(info, "Exporting TRK Detector layout to {}", fileName);
  std::ofstream conFile(fileName.c_str(), std::ios::out);
  conFile << "/// TRK configuration file: inn_radius  z_length  lay_thickness" << std::endl;
  for (const auto& layer : mLayers) {
    conFile << layer->getInnerRadius() << "\t" << layer->getZ() << "\t" << layer->getChipThickness() << std::endl;
  }
}

void Detector::configServices()
{
  mServices = TRKServices();
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

  // AIR
  float aAir[4] = {12.0107, 14.0067, 15.9994, 39.948};
  float zAir[4] = {6., 7., 8., 18.};
  float wAir[4] = {0.000124, 0.755267, 0.231781, 0.012827};
  float dAir = 1.20479E-3;

  // Carbon fiber
  float aCf[2] = {12.0107, 1.00794};
  float zCf[2] = {6., 1.};

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
    LOGP(fatal, "Could not find barrel volume while constructing TRK geometry");
  }
  new TGeoVolumeAssembly(GeometryTGeo::getTRKVolPattern());
  TGeoVolume* vTRK = geoManager->GetVolume(GeometryTGeo::getTRKVolPattern());
  vALIC->AddNode(vTRK, 2, new TGeoTranslation(0, 30., 0));

  char vstrng[100] = "TRKVol";
  vTRK->SetTitle(vstrng);

  for (auto& layer : mLayers) {
    layer->createLayer(vTRK);
  }

  // Add service for inner tracker
  mServices.createServices(vTRK);

  // Build the VD using the petal builder
  // Choose the VD design based on TRKBaseParam.layoutVD
  auto& trkPars = TRKBaseParam::Instance();

  o2::trk::clearVDSensorRegistry();

  switch (trkPars.layoutVD) {
    case kIRIS4:
      LOG(info) << "Building VD with IRIS4 layout";
      o2::trk::createIRIS4Geometry(vTRK);
      break;
    case kIRISFullCyl:
      LOG(info) << "Building VD with IRIS fully cylindrical layout";
      o2::trk::createIRISGeometryFullCyl(vTRK);
      break;
    case kIRISFullCyl3InclinedWalls:
      LOG(info) << "Building VD with IRIS fully cylindrical layout with 3 inclined walls";
      o2::trk::createIRISGeometry3InclinedWalls(vTRK);
      break;
    case kIRIS5:
      LOG(info) << "Building VD with IRIS5 layout";
      o2::trk::createIRIS5Geometry(vTRK);
      break;
    case kIRIS4a:
      LOG(info) << "Building VD with IRIS4a layout";
      o2::trk::createIRIS4aGeometry(vTRK);
      break;
    default:
      LOG(fatal) << "Unknown VD layout option: " << static_cast<int>(trkPars.layoutVD);
      break;
  }

  // Fill sensor names from registry right after geometry creation
  const auto& regs = o2::trk::vdSensorRegistry();
  mNumberOfVolumesVD = static_cast<int>(regs.size());
  mNumberOfVolumes = mNumberOfVolumesVD + mLayers.size();
  mSensorName.resize(mNumberOfVolumes);

  // Fill VD sensor names from registry
  int VDvolume = 0;
  for (const auto& sensor : regs) {
    mSensorName[VDvolume] = sensor.name;
    VDvolume++;
  }

  // Add MLOT sensor names
  for (int i = 0; i < mLayers.size(); i++) {
    mSensorName[VDvolume++].Form("%s%d", GeometryTGeo::getTRKSensorPattern(), i);
  }

  for (auto vd : mSensorName) {
    std::cout << "Volume name: " << vd << std::endl;
  }

  mServices.excavateFromVacuum("IRIS_CUTOUTsh");
  mServices.registerVacuum(vTRK);
}

void Detector::InitializeO2Detector()
{
  LOG(info) << "Initialize TRK O2Detector";
  mGeometryTGeo = GeometryTGeo::Instance();
  defineSensitiveVolumes();

  mSensorID.resize(mNumberOfVolumes); // hardcoded. TODO: change size when a different namingh scheme for VD is in place. Ideally could be 4 petals + 8 layers = 12
  for (int i = 0; i < mNumberOfVolumes; i++) {
    mSensorID[i] = gMC ? TVirtualMC::GetMC()->VolId(mSensorName[i]) : 0; // Volume ID from the Geant geometry
    LOGP(info, "{}: mSensorID={}, mSensorName={}", i, mSensorID[i], mSensorName[i].Data());
  }
}

void Detector::defineSensitiveVolumes()
{
  TGeoManager* geoManager = gGeoManager;
  TGeoVolume* v;

  TString volumeName;
  LOGP(info, "Adding TRK Sensitive Volumes");

  // Register VD sensors created by VDGeometryBuilder
  for (const auto& s : o2::trk::vdSensorRegistry()) {
    TGeoVolume* v = gGeoManager->GetVolume(s.name.c_str());
    if (!v) {
      LOGP(warning, "VD sensor volume '{}' not found", s.name);
      continue;
    }
    LOGP(info, "Adding VD Sensitive Volume {}", v->GetName());
    AddSensitiveVolume(v);
    // Optionally track first/last layers for TR references:
    if (s.region == o2::trk::VDSensorDesc::Region::Barrel && (s.idx == 0 /*innermost*/)) {
      mFirstOrLastLayers.push_back(s.name);
    }
  }

  // The names of the TRK sensitive volumes have the format: TRKLayer(0...mLayers.size()-1)
  for (int j{0}; j < mLayers.size(); j++) {
    volumeName = GeometryTGeo::getTRKSensorPattern() + TString::Itoa(j, 10);
    if (j == mLayers.size() - 1) {
      mFirstOrLastLayers.push_back(volumeName.Data());
    }
    LOGP(info, "Trying {}", volumeName.Data());
    v = geoManager->GetVolume(volumeName.Data());
    LOGP(info, "Adding TRK Sensitive Volume {}", v->GetName());
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

bool Detector::InsideFirstOrLastLayer(std::string layerName)
{
  bool inside = false;
  for (auto& firstOrLastLayer : mFirstOrLastLayers) {
    if (firstOrLastLayer == layerName) {
      inside = true;
      break;
    }
  }
  return inside;
}

bool Detector::ProcessHits(FairVolume* vol)
{
  // This method is called from the MC stepping
  if (!(fMC->TrackCharge())) {
    return false;
  }

  int subDetID = -1;
  int layer = -1;
  int volume = 0;
  int volID = vol->getMCid();

  bool notSens = false;
  while ((volume < mNumberOfVolumes) && (notSens = (volID != mSensorID[volume]))) {
    ++volume; /// there are 44 volumes, 36 for the VD (1 for each sensing element) and 8 for the MLOT (1 for each layer)
  }

  if (notSens) {
    return kFALSE; // RS: can this happen? This method must be called for sensors only?
  }

  if (volume < mNumberOfVolumesVD) {
    subDetID = 0; // VD. For the moment each "chip" is a volume./// TODO: change this logic once the naming scheme is changed
  } else {
    subDetID = 1; // MLOT
    layer = volume - mNumberOfVolumesVD;
  }

  // Is it needed to keep a track reference when the outer ITS volume is encountered?
  auto stack = (o2::data::Stack*)fMC->GetStack();
  // if (fMC->IsTrackExiting() && (lay == 0 || lay == mLayers.size() - 1)) {
  if (fMC->IsTrackExiting() && InsideFirstOrLastLayer(vol->GetName())) {
    // Keep the track refs for the innermost and outermost layers only
    o2::TrackReference tr(*fMC, GetDetId());
    tr.setTrackID(stack->GetCurrentTrackNumber());
    tr.setUserId(volume);
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
    int stave(0), halfstave(0), mod(0), chip(0);

    auto& trkPars = TRKBaseParam::Instance();

    if (subDetID == 1) {
      if (trkPars.layoutMLOT == o2::trk::eMLOTLayout::kSegmented) {
        fMC->CurrentVolOffID(1, chip);
        fMC->CurrentVolOffID(2, mod);
        if (mGeometryTGeo->getNumberOfHalfStaves(layer) == 2) {
          fMC->CurrentVolOffID(3, halfstave);
          fMC->CurrentVolOffID(4, stave);
        } else if (mGeometryTGeo->getNumberOfHalfStaves(layer) == 1) {
          fMC->CurrentVolOffID(3, stave);
        } else {
          LOGP(fatal, "Wrong number of halfstaves for layer {}", layer);
        }
      }
    } /// if VD, for the moment the volume is the "chipID" so no need to retrieve other elments

    unsigned short chipID = mGeometryTGeo->getChipIndex(subDetID, volume, layer, stave, halfstave, mod, chip);

    // Print(vol, volume, subDetID, layer, stave, halfstave, mod, chip, chipID);

    // mGeometryTGeo->Print();

    Hit* p = addHit(stack->GetCurrentTrackNumber(), chipID, mTrackData.mPositionStart.Vect(), positionStop.Vect(),
                    mTrackData.mMomentumStart.Vect(), mTrackData.mMomentumStart.E(), positionStop.T(),
                    mTrackData.mEnergyLoss, mTrackData.mTrkStatusStart, status);
    // p->SetTotalEnergy(vmc->Etot());

    // RS: not sure this is needed
    // Increment number of Detector det points in TParticle
    stack->addHit(GetDetId());
  }

  return true;
}

o2::trk::Hit* Detector::addHit(int trackID, unsigned short detID, const TVector3& startPos, const TVector3& endPos,
                               const TVector3& startMom, double startE, double endTime, double eLoss, unsigned char startStatus,
                               unsigned char endStatus)
{
  mHits->emplace_back(trackID, detID, startPos, endPos, startMom, startE, endTime, eLoss, startStatus, endStatus);
  return &(mHits->back());
}

void Detector::Print(FairVolume* vol, int volume, int subDetID, int layer, int stave, int halfstave, int mod, int chip, int chipID) const
{
  int currentVol(0);
  LOG(info) << "Current volume name: " << fMC->CurrentVolName() << " and ID " << fMC->CurrentVolID(currentVol);
  LOG(info) << "volume: " << volume << "/" << mNumberOfVolumes - 1;

  auto& trkPars = TRKBaseParam::Instance();

  if (subDetID == 1) { // MLOT
    if (trkPars.layoutMLOT == o2::trk::eMLOTLayout::kCylindrical) {
      LOG(info) << "off volume name 1 " << fMC->CurrentVolOffName(1) << "  chip: " << chip;
      LOG(info) << "SubDetector ID: " << subDetID << "  Layer: " << layer << "  Chip ID: " << chipID;
    } else {
      LOG(info) << "off volume name 1 " << fMC->CurrentVolOffName(1) << "  chip: " << chip;
      LOG(info) << "off volume name 2  " << fMC->CurrentVolOffName(2) << "  module: " << mod;
      if (mGeometryTGeo->getNumberOfHalfStaves(layer) == 2) { // staggered geometry
        LOG(info) << "off volume name 3  " << fMC->CurrentVolOffName(3) << "  halfstave: " << halfstave;
        LOG(info) << "off volume name 4  " << fMC->CurrentVolOffName(4) << "  stave: " << stave;
        LOG(info) << "SubDetector ID: " << subDetID << "  Layer: " << layer << "  staveinLayer: " << stave << "  Chip ID: " << chipID;
      } else if (mGeometryTGeo->getNumberOfHalfStaves(layer) == 1) { // turbo geometry
        LOG(info) << "off volume name 3  " << fMC->CurrentVolOffName(3) << "  stave: " << stave;
        LOG(info) << "SubDetector ID: " << subDetID << "  Layer: " << layer << "  staveinLayer: " << stave << "  Chip ID: " << chipID;
      }
    }
  } else {
    // VD
    LOG(info) << "SubDetector ID: " << subDetID << "  Chip ID: " << chipID;
  }

  LOG(info);
}

} // namespace trk
} // namespace o2

ClassImp(o2::trk::Detector);

// Define Factory method for calling from the outside
extern "C" {
o2::base::Detector* create_detector_trk(bool active)
{
  return o2::trk::Detector::create(active);
}
}
