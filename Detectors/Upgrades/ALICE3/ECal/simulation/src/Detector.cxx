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
/// \brief ECal geometry creation and hit processing
///
/// \author Evgeny Kryshen <evgeny.kryshen@cern.ch>

#include <FairVolume.h>
#include <TVirtualMC.h>
#include <TVirtualMCStack.h>
#include <TGeoManager.h>
#include <TGeoVolume.h>
#include <TGeoTube.h>
#include <TGeoArb8.h>
#include <TGeoTrd2.h>
#include <Math/Point2D.h>
#include <Math/Vector2D.h>
#include <DetectorsBase/Stack.h>
#include <ECalBase/Hit.h>
#include <ECalBase/ECalBaseParam.h>
#include <ECalBase/Geometry.h>
#include <ECalSimulation/Detector.h>

using o2::ecal::Hit;

namespace o2
{
namespace ecal
{
Detector::Detector(bool active)
  : o2::base::DetImpl<Detector>("ECL", active),
    mHits(o2::utils::createSimVector<o2::ecal::Hit>())
{
}

//==============================================================================
Detector::~Detector()
{
  if (mHits) {
    o2::utils::freeSimVector(mHits);
  }
}

//==============================================================================
void Detector::ConstructGeometry()
{
  createMaterials();
  createGeometry();
  defineSamplingFactor();
}

//==============================================================================
void Detector::defineSamplingFactor()
{
  TString mcname = TVirtualMC::GetMC()->GetName();
  TString mctitle = TVirtualMC::GetMC()->GetTitle();
  LOGP(info, "Defining sampling factor for mc={}' and title='{}'", mcname.Data(), mctitle.Data());
  if (mcname.Contains("Geant3")) {
    mSamplingFactorTransportModel = 0.983;
  } else if (mcname.Contains("Geant4")) {
    mSamplingFactorTransportModel = 1.;
  }
}

//==============================================================================
void Detector::createMaterials()
{
  LOGP(info, "Creating materials for ECL");

  // Air
  float aAir[4] = {12.0107, 14.0067, 15.9994, 39.948};
  float zAir[4] = {6., 7., 8., 18.};
  float wAir[4] = {0.000124, 0.755267, 0.231781, 0.012827};
  float dAir = 1.20479E-3;
  Mixture(0, "Air", aAir, zAir, dAir, 4, wAir);

  // Pb
  Material(1, "Pb", 207.2, 82, 11.35, 0.56, 0., nullptr, 0);

  // Polysterene scintillator (CH)
  float aP[2] = {12.011, 1.00794};
  float zP[2] = {6.0, 1.0};
  float wP[2] = {1.0, 1.0};
  float dP = 1.032;
  Mixture(2, "Scintillator", aP, zP, dP, -2, wP);

  // Al
  Material(3, "Al", 26.98, 13., 2.7, 8.9, 999., nullptr, 0);

  // PWO crystals
  float aX[3] = {207.19, 183.85, 16.0};
  float zX[3] = {82.0, 74.0, 8.0};
  float wX[3] = {1.0, 1.0, 4.0};
  float dX = 8.28;
  Mixture(4, "PbWO4", aX, zX, dX, -3, wX);

  int ifield = 2;     // magnetic field flag
  float fieldm = 10.; // maximum field value (in Kilogauss)
  float deemax = 0.1; // maximum fractional energy loss in one step (0 < deemax <=1)
  float tmaxfd = 10.; // maximum angle due to field permitted in one step (in degrees)
  o2::base::Detector::initFieldTrackingParams(ifield, fieldm);
  Medium(0, "Air", 0, 0, ifield, fieldm, tmaxfd, 1.0, deemax, 0.1, 10.0, nullptr, 0);
  Medium(1, "Pb", 1, 0, ifield, fieldm, tmaxfd, 0.1, deemax, 0.1, 0.1, nullptr, 0);
  Medium(2, "Scintillator", 2, 1, ifield, fieldm, tmaxfd, 0.001, deemax, 0.001, 0.001, nullptr, 0);
  Medium(3, "Al", 3, 0, ifield, fieldm, tmaxfd, 0.1, deemax, 0.001, 0.001, nullptr, 0);
  Medium(4, "Crystal", 4, 1, ifield, fieldm, tmaxfd, 0.1, deemax, 0.1, 0.1, nullptr, 0);
}

//==============================================================================
void Detector::InitializeO2Detector()
{
  LOG(info) << "Initialize ECal O2Detector";
  mGeometryTGeo = GeometryTGeo::Instance();
}

//==============================================================================
void Detector::Reset()
{
  if (!o2::utils::ShmManager::Instance().isOperational()) {
    mHits->clear();
  }
  mSuperParentIndices.clear();
  currentTrackId = -1;
  superparentId = -1;
}

//==============================================================================
void Detector::Register()
{
  // This will create a branch in the output tree called Hit, setting the last
  // parameter to kFALSE means that this collection will not be written to the file,
  // it will exist only during the simulation
  LOGP(info, "Registering hits");
  if (FairRootManager::Instance()) {
    FairRootManager::Instance()->RegisterAny(addNameTo("Hit").data(), mHits, true);
  }
}

//==============================================================================
void Detector::createGeometry()
{
  LOGP(info, "Creating ECal geometry");

  TGeoVolume* vALIC = gGeoManager->GetVolume("barrel");
  if (!vALIC) {
    LOGP(fatal, "Could not find barrel volume while constructing ECal geometry");
  }
  new TGeoVolumeAssembly(GeometryTGeo::getECalVolPattern());
  TGeoVolume* vECal = gGeoManager->GetVolume(GeometryTGeo::getECalVolPattern());
  vALIC->AddNode(vECal, 2, new TGeoTranslation(0, 30., 0));
  vECal->SetTitle("ECalVol");

  TGeoMedium* medAir = gGeoManager->GetMedium("ECL_Air");
  TGeoMedium* medPb = gGeoManager->GetMedium("ECL_Pb");
  TGeoMedium* medAl = gGeoManager->GetMedium("ECL_Al");
  TGeoMedium* medSc = gGeoManager->GetMedium("ECL_Scintillator");
  TGeoMedium* medPWO = gGeoManager->GetMedium("ECL_Crystal");

  // Get relevant parameters
  auto& pars = ECalBaseParam::Instance();
  auto& geo = Geometry::instance();

  double rMin = pars.rMin;
  double rMax = pars.rMax;
  double layerThickness = pars.pbLayerThickness + pars.scLayerThickness;
  double samplingModL = pars.frontPlateThickness + layerThickness * pars.nSamplingLayers - pars.pbLayerThickness;
  double crystalAlpha = geo.getCrystalAlpha();
  double samplingAlpha = geo.getSamplingAlpha();
  double tanCrystalAlpha = std::tan(crystalAlpha);
  double tanSamplingAlpha = std::tan(samplingAlpha);

  double sectorL = rMax - rMin;
  double crystalThetaMin = geo.getFrontFaceCenterTheta(pars.nCrystalModulesZ - 1) - crystalAlpha;
  double crystalHlMin = geo.getFrontFaceZatMinR(pars.nCrystalModulesZ - 1);
  double crystalHlMax = crystalHlMin + sectorL / std::tan(crystalThetaMin);
  double crystalHwMin = geo.getCrystalModW() / 2.;
  double crystalHwMax = crystalHwMin * rMax / rMin;
  auto crystalSectorShape = new TGeoTrap(sectorL / 2., 0, 0, crystalHlMin, crystalHwMin, crystalHwMin, 0, crystalHlMax, crystalHwMax, crystalHwMax, 0);
  auto crystalSectorVolume = new TGeoVolume("crystalSectorVolume", crystalSectorShape, medAir);
  AddSensitiveVolume(crystalSectorVolume);
  crystalSectorVolume->SetLineColor(kCyan + 1);
  crystalSectorVolume->SetTransparency(0);

  double samplingThetaAtMinZ = geo.getFrontFaceCenterTheta(pars.nCrystalModulesZ) + samplingAlpha;
  double samplingThetaAtMaxZ = geo.getFrontFaceCenterTheta(pars.nCrystalModulesZ + pars.nSamplingModulesZ - 1) - samplingAlpha;
  double samplingMinZatMinR = geo.getFrontFaceZatMinR(pars.nCrystalModulesZ) - geo.getSamplingModW() / std::sin(samplingThetaAtMinZ);
  double samplingMaxZatMinR = geo.getFrontFaceZatMinR(pars.nCrystalModulesZ + pars.nSamplingModulesZ - 1);
  double samplingMinZatMaxR = samplingMinZatMinR + sectorL / std::tan(samplingThetaAtMinZ);
  double samplingMaxZatMaxR = samplingMaxZatMinR + sectorL / std::tan(samplingThetaAtMaxZ);
  double hlMin = (samplingMaxZatMinR - samplingMinZatMinR) / 2.;
  double hlMax = (samplingMaxZatMaxR - samplingMinZatMaxR) / 2.;
  double zCenterMin = (samplingMaxZatMinR + samplingMinZatMinR) / 2.;
  double zCenterMax = (samplingMaxZatMaxR + samplingMinZatMaxR) / 2.;
  double zCenter = (zCenterMax + zCenterMin) / 2.;
  double thetaCenter = std::atan((zCenterMax - zCenterMin) / sectorL) * TMath::RadToDeg();
  double samplingHwMin = geo.getSamplingModW() / 2.;
  double samplingHwMax = samplingHwMin * rMax / rMin;
  auto samplingSectorShape = new TGeoTrap(sectorL / 2., thetaCenter, 90, hlMin, samplingHwMin, samplingHwMin, 0, hlMax, samplingHwMax, samplingHwMax, 0);
  auto samplingSectorVolume = new TGeoVolume("samplingSectorVolume", samplingSectorShape, medAir);
  AddSensitiveVolume(samplingSectorVolume);
  samplingSectorVolume->SetLineColor(kBlue + 1);
  samplingSectorVolume->SetTransparency(0);

  double sectorR = rMin + sectorL / 2.;
  for (int ism = 0; ism < pars.nSuperModules; ism++) {
    // crystal
    for (int i = 0; i < pars.nCrystalModulesPhi; i++) {
      int row = ism * pars.nCrystalModulesPhi + i;
      double phi0 = geo.getFrontFaceCenterCrystalPhi(row);
      double x = sectorR * std::cos(phi0);
      double y = sectorR * std::sin(phi0);
      auto rot = new TGeoRotation(Form("ecalcrystalsecrot%d", row), 90 + phi0 * TMath::RadToDeg(), 90, 0);
      vECal->AddNode(crystalSectorVolume, row, new TGeoCombiTrans(x, y, 0., rot));
    }
    // sampling
    for (int i = 0; i < pars.nSamplingModulesPhi; i++) {
      int row = ism * pars.nSamplingModulesPhi + i;
      double phi0 = geo.getFrontFaceCenterSamplingPhi(row);
      double x = sectorR * std::cos(phi0);
      double y = sectorR * std::sin(phi0);
      auto rot1 = new TGeoRotation(Form("ecalsamplingsec1rot%d", row), 90 + phi0 * TMath::RadToDeg(), 90, 0.);
      auto rot2 = new TGeoRotation(Form("ecalsamplingsec2rot%d", row), 90 + phi0 * TMath::RadToDeg(), 90, 180);
      vECal->AddNode(samplingSectorVolume, 2 * row + 0, new TGeoCombiTrans(x, y, zCenter, rot1));
      vECal->AddNode(samplingSectorVolume, 2 * row + 1, new TGeoCombiTrans(x, y, -zCenter, rot2));
    }
  }

  for (int m = 0; m < pars.nCrystalModulesZ; m++) {
    double tanBeta = geo.getTanBeta(m);
    double dx1 = crystalHwMin;
    double dx2 = crystalHwMin + pars.crystalModuleLength * tanCrystalAlpha;
    double dy1 = crystalHwMin;
    double dy2 = crystalHwMin + pars.crystalModuleLength * tanBeta;
    double dz = pars.crystalModuleLength / 2.;
    auto crystalModuleShape = new TGeoTrd2(dx1, dx2, dy1, dy2, dz);
    auto crystalModuleVolume = new TGeoVolume(Form("crystalmodule%d", m), crystalModuleShape, medPWO);
    AddSensitiveVolume(crystalModuleVolume);
    crystalModuleVolume->SetLineColor(kCyan + 1);
    crystalModuleVolume->SetTransparency(0);
    double theta = geo.getFrontFaceCenterTheta(m);
    double r = geo.getFrontFaceCenterR(m);
    double z = geo.getFrontFaceCenterZ(m);
    ROOT::Math::XYPoint pFrontFace(z, r - sectorR);
    ROOT::Math::Polar2DVector vFrontFaceToCenter(dz, theta);
    ROOT::Math::XYPoint pc = pFrontFace + vFrontFaceToCenter;
    auto rot1 = new TGeoRotation(Form("ecalcrystalrot%d", 2 * m), 0, 270 + theta * TMath::RadToDeg(), 90);
    crystalSectorVolume->AddNode(crystalModuleVolume, 2 * m, new TGeoCombiTrans(0, pc.x(), pc.y(), rot1));
    auto rot2 = new TGeoRotation(Form("ecalcrystalrot%d", 2 * m + 1), 0, 90 - theta * TMath::RadToDeg(), 90);
    crystalSectorVolume->AddNode(crystalModuleVolume, 2 * m + 1, new TGeoCombiTrans(0, -pc.x(), pc.y(), rot2));
  }

  for (int m = 0; m < pars.nSamplingModulesZ; m++) {
    int k = pars.nCrystalModulesZ + m;
    double tanBeta = geo.getTanBeta(k);
    double dx1 = samplingHwMin;
    double dx2 = samplingHwMin + samplingModL * tanSamplingAlpha;
    double dy1 = samplingHwMin;
    double dy2 = samplingHwMin + samplingModL * tanBeta;
    double dz = samplingModL / 2.;
    auto samplingModuleShape = new TGeoTrd2(dx1, dx2, dy1, dy2, dz);
    auto samplingModuleVolume = new TGeoVolume(Form("samplingmodule%d", m), samplingModuleShape, medSc);
    AddSensitiveVolume(samplingModuleVolume);
    samplingModuleVolume->SetLineColor(kAzure - 9);
    samplingModuleVolume->SetTransparency(0);
    double theta = geo.getFrontFaceCenterTheta(k);
    double r = geo.getFrontFaceCenterR(k);
    double z = geo.getFrontFaceCenterZ(k);
    ROOT::Math::XYPoint pFrontFace(z - zCenter, r - sectorR);
    ROOT::Math::Polar2DVector vFrontFaceToCenter(dz, theta);
    ROOT::Math::XYPoint pc = pFrontFace + vFrontFaceToCenter;
    auto rot1 = new TGeoRotation(Form("ecalsamplingrot%d", m), 0, 270 + theta * TMath::RadToDeg(), 90);
    samplingSectorVolume->AddNode(samplingModuleVolume, m, new TGeoCombiTrans(0, pc.x(), pc.y(), rot1));

    // adding front aluminium plate into the volume
    double fdx1 = dx1;
    double fdx2 = dx1 + pars.frontPlateThickness * tanSamplingAlpha;
    double fdy1 = dy1;
    double fdy2 = fdy1 + pars.frontPlateThickness * tanBeta;
    double fdz = pars.frontPlateThickness / 2.;
    auto frontShape = new TGeoTrd2(fdx1, fdx2, fdy1, fdy2, fdz);
    auto frontVolume = new TGeoVolume(Form("front%d", m), frontShape, medAl);
    samplingModuleVolume->AddNode(frontVolume, 0, new TGeoTranslation(0., 0., -dz + pars.frontPlateThickness / 2.));
    AddSensitiveVolume(frontVolume);
    frontVolume->SetLineColor(kAzure - 7);
    frontVolume->SetTransparency(0);

    // adding lead plates
    for (int i = 0; i < pars.nSamplingLayers - 1; i++) {
      double lz1 = pars.frontPlateThickness + pars.scLayerThickness + layerThickness * i;
      double lz2 = lz1 + pars.pbLayerThickness;
      double lzc = -dz + (lz1 + lz2) / 2.;
      double ldx1 = dx1 + lz1 * tanSamplingAlpha;
      double ldx2 = dx1 + lz2 * tanSamplingAlpha;
      double ldy1 = dy1 + lz1 * tanBeta;
      double ldy2 = dy1 + lz2 * tanBeta;
      double ldz = pars.pbLayerThickness / 2.;
      auto leadShape = new TGeoTrd2(ldx1, ldx2, ldy1, ldy2, ldz);
      auto leadVolume = new TGeoVolume(Form("lead%d_%d", m, i), leadShape, medPb);
      samplingModuleVolume->AddNode(leadVolume, i, new TGeoTranslation(0., 0., lzc));
      AddSensitiveVolume(leadVolume);
      leadVolume->SetLineColor(kAzure - 7);
      leadVolume->SetTransparency(0);
    }
  }

  if (pars.enableFwdEndcap) {
    // Build the ecal endcap
    TGeoTube* ecalEndcapShape = new TGeoTube("ECLECsh", 15.f, 160.f, 0.5 * (rMax - rMin));
    TGeoVolume* ecalEndcapVol = new TGeoVolume("ECLEC", ecalEndcapShape, medPb);
    ecalEndcapVol->SetLineColor(kAzure - 9);
    ecalEndcapVol->SetTransparency(0);
    vECal->AddNode(ecalEndcapVol, 1, new TGeoTranslation(0, 0, -450.f));
  }
  // gGeoManager->CloseGeometry();
  // gGeoManager->CheckOverlaps(0.0001);
}

//==============================================================================
bool Detector::ProcessHits(FairVolume* vol)
{
  LOGP(debug, "Processing hits");
  auto stack = (o2::data::Stack*)fMC->GetStack();
  int trackId = stack->GetCurrentTrackNumber();
  int parentId = stack->GetCurrentParentTrackNumber();

  if (trackId != currentTrackId) {
    auto superparentIndexIt = mSuperParentIndices.find(parentId);
    if (superparentIndexIt != mSuperParentIndices.end()) {
      superparentId = superparentIndexIt->second;
      mSuperParentIndices[trackId] = superparentIndexIt->second;
    } else {
      // for new incoming tracks the superparent index is equal to the track ID (for recursion)
      mSuperParentIndices[trackId] = trackId;
      superparentId = trackId;
    }
    currentTrackId = trackId;
  }

  double eloss = fMC->Edep();
  if (eloss < DBL_EPSILON) {
    return false;
  }

  TString volName = vol->GetName();
  bool isCrystal = volName.Contains("crystalmodule");
  bool isSampling = volName.Contains("samplingmodule");

  if (!isCrystal && !isSampling) {
    return false;
  }

  if (isCrystal) {
    LOGP(debug, "Processing crystal {}", volName.Data());
  } else {
    eloss *= mSamplingFactorTransportModel;
    LOGP(debug, "Processing scintillator {}", volName.Data());
  }
  int sectorId, moduleId;
  fMC->CurrentVolID(moduleId);
  fMC->CurrentVolOffID(1, sectorId);
  int cellID = Geometry::instance().getCellID(moduleId, sectorId, isCrystal);
  LOGP(debug, "isCrystal={} sectorId={} moduleId={} cellID={} eloss={}", isCrystal, sectorId, moduleId, cellID, eloss);

  int trackID = superparentId;
  auto hit = std::find_if(mHits->begin(), mHits->end(), [cellID, trackID](const Hit& hit) { return hit.GetTrackID() == trackID && hit.GetCellID() == cellID; });
  if (hit == mHits->end()) {
    float posX, posY, posZ, momX, momY, momZ, energy;
    fMC->TrackPosition(posX, posY, posZ);
    fMC->TrackMomentum(momX, momY, momZ, energy);
    auto pos = math_utils::Point3D<float>(posX, posY, posZ);
    auto mom = math_utils::Vector3D<float>(momX, momY, momZ);
    float time = fMC->TrackTime() * 1e9; // time in ns
    mHits->emplace_back(trackID, cellID, pos, mom, time, eloss);
    stack->addHit(GetDetId());
  } else {
    hit->SetEnergyLoss(hit->GetEnergyLoss() + eloss);
  }
  return true;
}

} // namespace ecal
} // namespace o2

ClassImp(o2::ecal::Detector);
