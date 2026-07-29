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

#include "TRKSimulation/Detector.h"

#include "DetectorsBase/Stack.h"

#include "TRKBase/Specs.h"
#include "TRKBase/TRKBaseParam.h"
#include "TRKSimulation/Hit.h"
#include "TRKSimulation/VDGeometryBuilder.h"
#include "TRKSimulation/VDSensorRegistry.h"
#include <TGeoVolume.h>
#include <TVirtualMC.h>
#include <TVirtualMCStack.h>

#include <FairVolume.h>

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
    if (!trkPars.disableFT3) {
      configFT3ScopingV3();
    }
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
      const std::vector<float> length{127.985f, 127.985f, 127.985f, 127.985f, 127.985f, 255.9f, 255.9f, 255.9f};
      LOGP(warning, "Loading cylindrical configuration for ALICE3 TRK");
      for (int i{0}; i < constants::ML::nLayers + constants::OT::nLayers; ++i) {
        std::string name = GeometryTGeo::getTRKLayerPattern() + std::to_string(i);
        mLayers.push_back(std::make_unique<TRKCylindricalLayer>(i, name, rInn[i], length[i], thick, MatBudgetParamMode::Thickness));
      }
      break;
    }
    case kSegmented: {
      const std::vector<float> tiltAngles{11.2f, 11.9f, 11.4f, 0.f, 0.f, 0.f, 0.f, 0.f};
      // const std::vector<float> tiltAngles{10.f, 16.1f, 19.2f, 0.f, 0.f, 0.f, 0.f, 0.f};
      const std::vector<int> nStaves{10, 14, 18, 26, 38, 32, 42, 56};
      // const std::vector<int> nStaves{10, 16, 22, 26, 38, 32, 42, 56};
      const std::vector<int> nMods{11, 11, 11, 11, 11, 22, 22, 22};

      const std::vector<float> stagOffsets{0.f, 0.f, 0.f, 1.17f, 0.89f};

      LOGP(warning, "Loading segmented configuration for ALICE3 TRK");
      for (int i{0}; i < constants::ML::nLayers + constants::OT::nLayers; ++i) {
        std::string name = GeometryTGeo::getTRKLayerPattern() + std::to_string(i);
        if (i < constants::ML::nLayers) {
          mLayers.push_back(std::make_unique<TRKMLLayer>(i, name, rInn[i], stagOffsets[i], tiltAngles[i], nStaves[i], nMods[i], thick, MatBudgetParamMode::Thickness));
        } else {
          mLayers.push_back(std::make_unique<TRKOTLayer>(i, name, rInn[i], tiltAngles[i], nStaves[i], nMods[i], thick, MatBudgetParamMode::Thickness));
        }
      }
      break;
    }
    default:
      LOGP(fatal, "Unknown option {} for configMLOT", static_cast<int>(trkPars.layoutMLOT));
      break;
  }
}

void Detector::configFT3ScopingV3()
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

  for (int direction : {kBackward, kForward}) {
    mFT3LayerName[direction].clear();
    const std::array<LayerConfig, numberOfLayers>& layerConfig = (direction == kBackward) ? layersConfigCSide : layersConfigASide;
    for (int layerNumber = 0; layerNumber < numberOfLayers; layerNumber++) {
      if (!enabled[layerNumber]) {
        continue;
      }
      const std::string directionName = std::to_string(direction);
      const std::string layerName = GeometryTGeo::getFT3LayerPattern() + directionName + std::string("_") + std::to_string(layerNumber);
      mFT3LayerName[direction].push_back(layerName.c_str());
      const float z = layerConfig[layerNumber][0];
      const float rIn = layerConfig[layerNumber][1];
      const float rOut = layerConfig[layerNumber][2];
      const float x0 = layerConfig[layerNumber][3];
      LOG(info) << "buildFT3ScopingV3 -> Adding Layer " << layerNumber << "/" << numberOfLayers << " " << layerName << " at z = " << z;
      // Add layers
      const bool isMiddleLayer = layerNumber < 3;
      auto& thisLayer = mFT3Layers[direction].emplace_back(direction, layerNumber, layerName, z, rIn, rOut, x0, isMiddleLayer);
    }
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
      case kCylindrical: {
        // Expected column mapping in the text file (separated by \t):
        // tmpBuff[0] = rInn
        // tmpBuff[1] = length
        // tmpBuff[2] = thick
        // tmpBuff[3] = matBudgetMode (optional, default = Thickness)

        // Cylindrical requires at least 3 parameters
        if (tmpBuff.size() < 3) {
          LOGP(fatal, "Invalid configuration for cylindrical layer {}: insufficient parameters.", layerCount);
        }

        float rInn = tmpBuff[0];
        float length = tmpBuff[1];
        float thick = tmpBuff[2];

        // Default mode is Thickness
        MatBudgetParamMode matBudgetMode = MatBudgetParamMode::Thickness;
        if (tmpBuff.size() >= 4) {
          matBudgetMode = static_cast<MatBudgetParamMode>(static_cast<int>(tmpBuff[3]));
        }

        mLayers.push_back(std::make_unique<TRKCylindricalLayer>(layerCount, name, rInn, length, thick, matBudgetMode));
        break;
      }
      case kSegmented: {
        // Expected column mapping in the text file (separated by \t):
        // tmpBuff[0] = rInn
        // tmpBuff[1] = thick
        // tmpBuff[2] = tiltAngle
        // tmpBuff[3] = nStaves
        // tmpBuff[4] = nMods
        // tmpBuff[5] = stagOffset (required ONLY for ML)
        // tmpBuff[6] = matBudgetMode (optional, default = Thickness)

        // Base parameters for all segmented layers (at least 5 needed)
        if (tmpBuff.size() < 5) {
          LOGP(fatal, "Invalid configuration for segmented layer {}: missing base parameters.", layerCount);
        }

        float rInn = tmpBuff[0];
        float thick = tmpBuff[1];
        float tiltAngle = tmpBuff[2];
        int nStaves = static_cast<int>(tmpBuff[3]);
        int nMods = static_cast<int>(tmpBuff[4]);

        // Default mode is Thickness
        MatBudgetParamMode matBudgetMode = MatBudgetParamMode::Thickness;

        if (layerCount < constants::ML::nLayers) {
          // ML layers require stagOffset (index 5)
          if (tmpBuff.size() < 6) {
            LOGP(fatal, "Invalid configuration for ML layer {}: stagOffset is missing.", layerCount);
          }
          float stagOffset = tmpBuff[5];

          if (tmpBuff.size() >= 7) {
            matBudgetMode = static_cast<MatBudgetParamMode>(static_cast<int>(tmpBuff[6]));
          }

          mLayers.push_back(std::make_unique<TRKMLLayer>(layerCount, name, rInn, stagOffset, tiltAngle, nStaves, nMods, thick, matBudgetMode));
        } else {
          // OT layers do NOT have stagOffset. The optional mode is at index 5.
          if (tmpBuff.size() >= 6) {
            matBudgetMode = static_cast<MatBudgetParamMode>(static_cast<int>(tmpBuff[5]));
          }

          mLayers.push_back(std::make_unique<TRKOTLayer>(layerCount, name, rInn, tiltAngle, nStaves, nMods, thick, matBudgetMode));
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

  // Place forward tracking discs

  TGeoVolume* A3IPvac = gGeoManager->GetVolume("OUT_PIPEVACUUM");
  if (!A3IPvac) {
    LOG(info) << "Running simulation with no beam pipe.";
  }

  // TODO: disambiquate layer/disk below
  // This will need to adapt to the new scheme
  if (!A3IPvac) {
    for (int direction : {kBackward, kForward}) { // Backward layers at mLayers[0]; Forward layers at mLayers[1]
      const std::string directionString = direction ? "Forward" : "Backward";
      LOG(info) << "  Creating FT3 without beampipe " << directionString << " layers:";
      for (int iLayer = 0; iLayer < mFT3Layers[direction].size(); iLayer++) {
        mFT3Layers[direction][iLayer].createLayer(vTRK);
      }
    }
  } else { // If beampipe is enabled append inner disks to beampipe filling volume, this should be temporary.
    TGeoVolume* volIFT3 = new TGeoVolumeAssembly(GeometryTGeo::getFT3InnerVolPattern());
    for (int direction : {kBackward, kForward}) {
      const std::string directionString = direction ? "Forward" : "Backward";
      LOG(info) << "  Creating FT3 " << directionString << " layers:";
      for (int iLayer = 0; iLayer < mFT3Layers[direction].size(); iLayer++) {
        LOG(info) << "  Creating " << directionString << " layer " << iLayer;
        if (mFT3Layers[direction][iLayer].getIsInMiddleLayer()) { // ML disks
          mFT3Layers[direction][iLayer].createLayer(volIFT3);
        } else {
          mFT3Layers[direction][iLayer].createLayer(vTRK);
        }
      }
    }
    A3IPvac->AddNode(volIFT3, 2, new TGeoTranslation(0., 0., 0.));
  }
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

  // Add FT3 sensitive volumes
  // TODO: do we need to loop over all volumes in our code, or can we use the geomanager?
  // Get the flat list of ALL volumes present in the geometry
  TObjArray* allVolumes = geoManager->GetListOfVolumes();
  int nVolumes = allVolumes->GetEntriesFast();

  LOG(info) << "Adding FT3 Sensitive Volumes by iterating over all geometry volumes...";

  for (int direction : {kBackward, kForward}) {
    for (int iLayer = 0; iLayer < getNumberOfFT3Layers(); iLayer++) {
      int iSens = 0;

      // Build the "signatures" (prefixes) of the names for the various layouts for this specific layer and direction:

      // 1. Trapezoidal/Cylindrical (format: FT3Sensor_<dir>_<layer>)
      std::string sig1 = Form("%s_%d_%d", GeometryTGeo::getFT3SensorPattern(), direction, iLayer);

      // 2. Segmented front/back (format: FT3Sensor_front_<layer>_<dir>_...)
      std::string sig2 = "FT3Sensor_front_" + std::to_string(iLayer) + "_" + std::to_string(direction);
      std::string sig3 = "FT3Sensor_back_" + std::to_string(iLayer) + "_" + std::to_string(direction);

      // 3. SegmentedStave (format: FT3Sensor_<dir>_<layer>_...)
      // Add the trailing underscore to avoid confusing it with sig1
      std::string sig4 = "FT3Sensor_Active_" + std::to_string(direction) + "_" + std::to_string(iLayer) + "_";

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
          /*
          int volID = gMC ? TVirtualMC::GetMC()->VolId(vName.c_str()) : 0;
          if (volID > 0) {
            mActiveSensorMap[volID] = iLayer;
          }
          */
          iSens++;
        }
      }

      if (iSens == 0) {
        LOG(error) << "NO sensitive volume found for FT3 direction " << direction << ", layer " << iLayer;
      } else {
        LOG(info) << iSens << " sensitive volume(s) added for FT3 direction " << direction << " layer " << iLayer;
      }
    }
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

  if (volume < mNumberOfVolumesVD) {
    subDetID = 0; // VD. For the moment each "chip" is a volume./// TODO: change this logic once the naming scheme is changed
  } else {
    subDetID = 1; // MLOT
    layer = volume - mNumberOfVolumesVD;
  }

  if (strstr(vol->GetName(), "FT3Sensor_Active") || strstr(vol->GetName(), "FT3Chip")) {
    subDetID = 2;
    notSens = false;
  }

  // TODO: add corresponding logic for disks. I think Ruben is right; this is only called for active volumes!
  if (notSens) {
    LOG(info) << "ProcessHit called for insensitive volume " << vol->GetName();
    return kFALSE; // RS: can this happen? This method must be called for sensors only?
  }

  // Is it needed to keep a track reference when the outer ITS volume is encountered?
  auto stack = (o2::data::Stack*)fMC->GetStack();
  // if (fMC->IsTrackExiting() && (lay == 0 || lay == mLayers.size() - 1)) {
  if (fMC->IsTrackExiting() && subDetID < 2 && InsideFirstOrLastLayer(vol->GetName())) {
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
    else if (subDetID == 2) {
      mGeometryTGeo->extractChipIdsFT3(vol->GetName(), layer, stave, chip);
    }
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
