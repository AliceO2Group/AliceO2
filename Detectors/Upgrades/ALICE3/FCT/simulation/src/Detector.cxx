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
#include "FCTBase/GeometryTGeo.h"
#include "FCTSimulation/Detector.h"
#include "FCTSimulation/FCTLayer.h"
#include "FCTBase/FCTBaseParam.h"

#include "SimulationDataFormat/Stack.h"
#include "SimulationDataFormat/TrackReference.h"

// FairRoot includes
#include "FairDetector.h"    // for FairDetector
#include <fairlogger/Logger.h> // for LOG, LOG_IF
#include "FairModule.h"      // for FairModule
#include "FairRootManager.h" // for FairRootManager
#include "FairRun.h"         // for FairRun
#include "FairRuntimeDb.h"   // for FairRuntimeDb
#include "FairVolume.h"      // for FairVolume
#include "FairRootManager.h"

#include "TGeoManager.h"     // for TGeoManager, gGeoManager
#include "TGeoMedium.h"      // for TGeoMedium
#include "TGeoTube.h"        // for TGeoTube
#include "TGeoPcon.h"        // for TGeoPcon
#include "TGeoVolume.h"      // for TGeoVolume, TGeoVolumeAssembly
#include "TParticle.h"       // for TParticle
#include "TString.h"         // for TString, operator+
#include "TVirtualMC.h"      // for gMC, TVirtualMC
#include "TVirtualMCStack.h" // for TVirtualMCStack
#include "TMath.h"           // for Abs, ATan, Exp, Tan
#include "TCanvas.h"
#include "TDatime.h" // for GetDate, GetTime

#include <cstdio> // for NULL, snprintf

using namespace o2::fct;
using o2::itsmft::Hit;

//_________________________________________________________________________________________________
Detector::Detector()
  : o2::base::DetImpl<Detector>("FCT", kTRUE),
    mTrackData(),
    mHits(o2::utils::createSimVector<o2::itsmft::Hit>())
{
}

//_________________________________________________________________________________________________
void Detector::buildFCTFromFile(std::string configFileName)
{
  // Geometry description from file. One line per disk
  // z_layer r_in r_out Layerx2X0
  // This simple file reader is not failproof. Do not add empty lines!

  /*
  # Sample MFT configuration
  # z_layer    r_in    r_out   Layerx2X0   layerType (0 disks, 1 squares)
  -45.3       2.5     9.26    0.0042      1
  -46.7       2.5     9.26    0.0042      1
  -48.6       2.5     9.8     0.0042      1
  -50.0       2.5     9.8     0.0042      1
  -52.4       2.5     10.43   0.0042      1
  -53.8       2.5     10.43   0.0042      1
  -67.7       3.82    13.01   0.0042      1
  -69.1       3.82    13.01   0.0042      1
  -76.1       3.92    14.35   0.0042      1
  -77.5       3.92    14.35   0.0042      1
  */

  mLayerName.clear();
  mLayers.clear();
  mConverterLayers.clear();
  mLayerID.clear();

  LOG(info) << "Building FCT Detector: From file";
  LOG(info) << "   FCT detector configuration: " << configFileName;
  std::ifstream ifs(configFileName.c_str());
  if (!ifs.good()) {
    LOG(fatal) << " Invalid FCTBase.configFile!";
  }
  std::string tempstr;
  float z_layer, r_in, r_out_l_side, Layerx2X0;
  char delimiter;
  int layerNumber = 0;
  int layerNumberSquare = 0;
  int layerNumberDisk = 0;
  int pseumin = 3;
  int pseumax = 5;
  int layerType;
  bool r_toggle = false;

  while (std::getline(ifs, tempstr)) {
    if (tempstr[0] == '#') {
      LOG(info) << " Comment: " << tempstr;
      int loc_r_toggle = tempstr.find("r_toggle");
      if (loc_r_toggle != -1) {
        r_toggle = true;
        LOG(info) << " Comment: R toggle activated";
      }
      continue;
    }
    std::istringstream iss(tempstr);
    iss >> layerType;
    iss >> z_layer;
    if (r_toggle) {
      auto pseurap_to_ang = [](float eta) { return 2. * TMath::ATan(TMath::Exp(-eta)); };
      r_in = TMath::Abs(z_layer) * TMath::Tan(pseurap_to_ang(pseumax));
      r_out_l_side = TMath::Abs(z_layer) * TMath::Tan(pseurap_to_ang(pseumin));
    } else {
      iss >> r_in;
      iss >> r_out_l_side;
    }
    iss >> Layerx2X0;

    std::string layerName = GeometryTGeo::getFCTLayerPattern() + std::string("_") + std::to_string(layerNumber);
    mLayerName.push_back(layerName);
    if (layerType == 0) {
      LOG(info) << "Adding Disk Layer " << layerName << " at z = " << z_layer << " ; r_in = " << r_in << " ; r_out = " << r_out_l_side << " x/X0 = " << Layerx2X0;
      layerNumberDisk++;
    } else if (layerType == 1) {
      LOG(info) << "Adding Square Layer " << layerName << " at z = " << z_layer << " ; r_in = " << r_in << " ; l_side = " << r_out_l_side << " x/X0 = " << Layerx2X0;
      layerNumberSquare++;
    } else if (layerType == 2) {
      LOG(info) << "Adding passive converter Layer " << layerName << " at z = " << z_layer << " ; r_in = " << r_in << " ; r_out = " << r_out_l_side << " x/X0 = " << Layerx2X0;
    }

    if (layerType == 0 || layerType == 1) {
      mLayers.emplace_back(layerNumber, layerName, z_layer, r_in, r_out_l_side, Layerx2X0, layerType);
    } else if (layerType == 2) {
      mConverterLayers.emplace_back(layerNumber, layerName, z_layer, r_in, r_out_l_side, Layerx2X0, layerType);
    }
    layerNumber++;
  }

  mNumberOfLayers = layerNumber;
  LOG(info) << " Loaded FCT Detector with  " << mNumberOfLayers << " layers";
  LOG(info) << " Of which " << layerNumberDisk << " are disks";
  LOG(info) << " Of which " << layerNumberSquare << " are disks";
}

//_________________________________________________________________________________________________
void Detector::exportLayout()
{
  // Export FCT Layout description to file. One line per disk
  // z_layer r_in r_out Layerx2X0

  TDatime* time = new TDatime();
  TString configFileName = "FCT_layout_";
  configFileName.Append(Form("%d.cfg", time->GetTime()));

  // std::string configFileName = "FCT_layout.cfg";

  LOG(info) << "Exporting FCT Detector layout to " << configFileName;

  std::ofstream fOut(configFileName, std::ios::out);
  if (!fOut) {
    printf("Cannot open file\n");
    return;
  }
  fOut << "#   z_layer   r_in   r_out_l_side   Layerx2X0" << std::endl;
  for (auto layer : mLayers) {
    if (layer.getType() == 0) {
      fOut << layer.getZ() << "  " << layer.getInnerRadius() << "  " << layer.getOuterRadius() << "  " << layer.getx2X0() << std::endl;
    } else if (layer.getType() == 1) {
      fOut << layer.getZ() << "  " << layer.getInnerRadius() << "  " << layer.getSideLength() << "  " << layer.getx2X0() << std::endl;
    }
  }
}

//_________________________________________________________________________________________________
void Detector::buildBasicFCT(const FCTBaseParam& param)
{
  // Build a basic parametrized FCT detector with nLayers equally spaced between z_first and z_first+z_length
  // Covering pseudo rapidity [etaIn,etaOut]. Silicon thinkness computed to match layer x/X0

  LOG(info) << "Building FCT Detector: Conical Telescope";

  auto z_first = param.z0;
  auto z_length = param.zLength;
  auto etaIn = param.etaIn;
  auto etaOut = param.etaOut;
  auto Layerx2X0 = param.Layerx2X0;
  mNumberOfLayers = param.nLayers;
  mLayerID.clear();

  Int_t type = 0; // Disk

  for (Int_t layerNumber = 0; layerNumber < mNumberOfLayers; layerNumber++) {
    std::string layerName = GeometryTGeo::getFCTLayerPattern() + std::to_string(layerNumber); // + mNumberOfLayers * direction);
    mLayerName.push_back(layerName);

    // Adds evenly spaced layers
    Float_t layerZ = z_first + (layerNumber * z_length / (mNumberOfLayers - 1)) * std::copysign(1, z_first);
    Float_t rIn = std::abs(layerZ * std::tan(2.f * std::atan(std::exp(-etaIn))));
    Float_t rOut = std::abs(layerZ * std::tan(2.f * std::atan(std::exp(-etaOut))));
    mLayers.emplace_back(layerNumber, layerName, layerZ, rIn, rOut, Layerx2X0, type);
  }
}

//_________________________________________________________________________________________________
void Detector::buildFCTV1()
{
  // Build default FCT detector

  LOG(info) << "Building FCT Detector: V1";

  mNumberOfLayers = 9;
  Float_t layersx2X0 = 1.e-2;

  std::vector<std::array<Float_t, 4>> layersConfig{
    {-442.0, 6.0, 44.1, layersx2X0}, // {z_layer, r_in, r_out, Layerx2X0}
    {-444.0, 6.0, 44.3, layersx2X0},
    {-446.0, 6.0, 44.5, layersx2X0},
    {-448.0, 6.0, 44.7, layersx2X0},
    {-450.0, 6.1, 44.9, layersx2X0},
    {-460.0, 6.2, 45.9, layersx2X0},
    {-470.0, 6.3, 46.9, layersx2X0},
    {-480.0, 6.5, 47.9, layersx2X0},
    {-490.0, 6.6, 48.9, layersx2X0}};

  mLayerID.clear();
  mLayerName.clear();
  mLayers.clear();

  Int_t type = 0; // Disk

  for (int layerNumber = 0; layerNumber < mNumberOfLayers; layerNumber++) {
    std::string layerName = GeometryTGeo::getFCTLayerPattern() + std::to_string(layerNumber);
    mLayerName.push_back(layerName);
    Float_t z = layersConfig[layerNumber][0];

    Float_t rIn = layersConfig[layerNumber][1];
    Float_t rOut = layersConfig[layerNumber][2];
    Float_t x0 = layersConfig[layerNumber][3];

    LOG(info) << "Adding Layer " << layerName << " at z = " << z;
    // Add layers
    mLayers.emplace_back(layerNumber, layerName, z, rIn, rOut, x0, type);
  }
}

//_________________________________________________________________________________________________
Detector::Detector(Bool_t active)
  : o2::base::DetImpl<Detector>("FCT", active),
    mTrackData(),
    mHits(o2::utils::createSimVector<o2::itsmft::Hit>())
{
}

//_________________________________________________________________________________________________
Detector::Detector(const Detector& rhs)
  : o2::base::DetImpl<Detector>(rhs),
    mTrackData(),

    /// Container for data points
    mHits(o2::utils::createSimVector<o2::itsmft::Hit>())
{
  mLayerID = rhs.mLayerID;
  mLayerName = rhs.mLayerName;
  mNumberOfLayers = rhs.mNumberOfLayers;
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

  mLayerID = rhs.mLayerID;
  mLayerName = rhs.mLayerName;
  mNumberOfLayers = rhs.mNumberOfLayers;
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
  LOG(info) << "Initialize FCT O2Detector";

  mGeometryTGeo = GeometryTGeo::Instance();

  defineSensitiveVolumes();
}

//_________________________________________________________________________________________________
Bool_t Detector::ProcessHits(FairVolume* vol)
{
  // This method is called from the MC stepping
  if (!(fMC->TrackCharge())) {
    return kFALSE;
  }

  Int_t lay = 0, volID = vol->getMCid();
  while ((lay <= mLayerID.size()) && (volID != mLayerID[lay])) {
    ++lay;
  }

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
    return kFALSE; // do nothing
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
  Int_t ifield = 2;
  Float_t fieldm = 10.0;
  o2::base::Detector::initFieldTrackingParams(ifield, fieldm);

  Float_t tmaxfdSi = 0.1;    // .10000E+01; // Degree
  Float_t stemaxSi = 0.0075; //  .10000E+01; // cm
  Float_t deemaxSi = 0.1;    // 0.30000E-02; // Fraction of particle's energy 0<deemax<=1
  Float_t epsilSi = 1.0E-4;  // .10000E+01;
  Float_t stminSi = 0.0;     // cm "Default value used"

  Float_t tmaxfdAir = 0.1;        // .10000E+01; // Degree
  Float_t stemaxAir = .10000E+01; // cm
  Float_t deemaxAir = 0.1;        // 0.30000E-02; // Fraction of particle's energy 0<deemax<=1
  Float_t epsilAir = 1.0E-4;      // .10000E+01;
  Float_t stminAir = 0.0;         // cm "Default value used"

  Float_t tmaxfdPb = 10.;  // .10000E+01; // Degree
  Float_t stemaxPb = 0.01; //  .10000E+01; // cm
  Float_t deemaxPb = 0.1;  // .10000E+01; // Fraction of particle's energy 0<deemax<=1
  Float_t epsilPb = 0.1;   // .10000E+01;
  Float_t stminPb = 0.0;   // cm "Default value used"

  // AIR
  Float_t aAir[4] = {12.0107, 14.0067, 15.9994, 39.948};
  Float_t zAir[4] = {6., 7., 8., 18.};
  Float_t wAir[4] = {0.000124, 0.755267, 0.231781, 0.012827};
  Float_t dAir = 1.20479E-3;

  o2::base::Detector::Mixture(1, "AIR$", aAir, zAir, dAir, 4, wAir);
  o2::base::Detector::Medium(1, "AIR$", 1, 0, ifield, fieldm, tmaxfdAir, stemaxAir, deemaxAir, epsilAir, stminAir);

  // Add Silicon
  o2::base::Detector::Material(3, "SI$", 0.28086E+02, 0.14000E+02, 0.23300E+01, 0.93600E+01, 0.99900E+03);
  o2::base::Detector::Medium(3, "SI$", 3, 0, ifield, fieldm, tmaxfdSi, stemaxSi, deemaxSi, epsilSi, stminSi);

  // Add Lead (copied from EMCAL)
  o2::base::Detector::Material(2, "Pb$", 207.2, 82, 11.35, 0.56, 0.);
  o2::base::Detector::Medium(2, "Pb$", 2, 0, ifield, fieldm, tmaxfdPb, stemaxPb, deemaxPb, epsilPb, stminPb);
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
  // FCT Base configuration parameters
  auto& fctBaseParam = FCTBaseParam::Instance();

  // Set the parameters for the detector layout
  if (fctBaseParam.configFile != "") {
    LOG(info) << "FCT Geometry configuration file provided. Overriding FCTBase.geoModel configuration.";
    buildFCTFromFile(fctBaseParam.configFile);
  } else {
    switch (fctBaseParam.geoModel) {
      case Default:
        buildFCTV1(); // FCTV1
        break;
      case Telescope:
        buildBasicFCT(fctBaseParam); // BasicFCT = Parametrized telescopic detector (equidistant layers)
        break;
      default:
        LOG(fatal) << "Invalid Geometry.\n";
        break;
    }
  }
  exportLayout();

  // Create detector materials
  createMaterials();

  // Construct the detector geometry
  createGeometry();
}

//_________________________________________________________________________________________________
void Detector::createGeometry()
{

  mGeometryTGeo = GeometryTGeo::Instance();

  TGeoVolume* volFCT = new TGeoVolumeAssembly(GeometryTGeo::getFCTVolPattern());
  TGeoVolume* volIFCT = new TGeoVolumeAssembly(GeometryTGeo::getFCTInnerVolPattern());

  LOG(info) << "GeometryBuilder::buildGeometry volume name = " << GeometryTGeo::getFCTVolPattern();

  TGeoVolume* vALIC = gGeoManager->GetVolume("barrel");
  if (!vALIC) {
    LOG(fatal) << "Could not find the top volume";
  }

  TGeoVolume* A3IPvac = gGeoManager->GetVolume("OUT_PIPEVACUUM");
  if (!A3IPvac) {
    LOG(info) << "Running simulation with no beam pipe.";
  }

  LOG(debug) << "FCT createGeometry: "
             << Form("gGeoManager name is %s title is %s", gGeoManager->GetName(), gGeoManager->GetTitle());

  LOG(info) << "Creating FCT layers:";
  if (A3IPvac) {
    for (Int_t iLayer = 0; iLayer < mLayers.size(); iLayer++) {
      if (std::abs(mLayers[iLayer].getZ()) < 25) {
        mLayers[iLayer].createLayer(volIFCT);
      } else {
        mLayers[iLayer].createLayer(volFCT);
      }
    }
    A3IPvac->AddNode(volIFCT, 2, new TGeoTranslation(0., 0., 0.));
    vALIC->AddNode(volFCT, 2, new TGeoTranslation(0., 30., 0.));
  } else {
    for (Int_t iLayer = 0; iLayer < mLayers.size(); iLayer++) {
      mLayers[iLayer].createLayer(volFCT);
    }
    for (Int_t iLayer = 0; iLayer < mConverterLayers.size(); iLayer++) {
      mConverterLayers[iLayer].createLayer(volFCT);
    }
    vALIC->AddNode(volFCT, 2, new TGeoTranslation(0., 30., 0.));
  }
  LOG(info) << "Registering FCT SensitiveLayerIDs:";
  for (int iLayer = 0; iLayer < mLayers.size(); iLayer++) {
    auto layerID = gMC ? TVirtualMC::GetMC()->VolId(Form("%s_%d", GeometryTGeo::getFCTSensorPattern(), mLayers[iLayer].getLayerNumber())) : 0;
    mLayerID.push_back(layerID);
    LOG(info) << "  mLayerID[" << mLayers[iLayer].getLayerNumber() << "] = " << layerID;
  }
}

//_________________________________________________________________________________________________
void Detector::defineSensitiveVolumes()
{
  TGeoManager* geoManager = gGeoManager;
  TGeoVolume* v;

  TString volumeName;
  LOG(info) << "Adding FCT Sensitive Volumes";

  // The names of the FCT sensitive volumes have the format: FCTSensor_(0,1)_(0...sNumberLayers-1)
  for (Int_t iLayer = 0; iLayer < mLayers.size(); iLayer++) {
    volumeName = o2::fct::GeometryTGeo::getFCTSensorPattern() + std::to_string(mLayers[iLayer].getLayerNumber());
    v = geoManager->GetVolume(Form("%s_%d", GeometryTGeo::getFCTSensorPattern(), mLayers[iLayer].getLayerNumber()));
    LOG(info) << "Adding FCT Sensitive Volume => " << v->GetName();
    AddSensitiveVolume(v);
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

ClassImp(o2::fct::Detector);
