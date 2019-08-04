// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include <DetectorsBase/Detector.h>
#include <DetectorsBase/MaterialManager.h>
#include <DetectorsPassive/Compensator.h>
#include <DetectorsPassive/HallSimParam.h>
#include <TGeoCompositeShape.h>
#include <TGeoManager.h>
#include <TGeoMaterial.h>
#include <TGeoMatrix.h>
#include <TGeoMedium.h>
#include <TGeoTube.h>
#include <TGeoVolume.h>
#include <TGeoXtru.h>
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>

using namespace o2::passive;

Compensator::~Compensator() = default;

Compensator::Compensator() : FairModule("Compensator", "") {}
Compensator::Compensator(const char* name, const char* Title) : FairModule(name, Title) {}
Compensator::Compensator(const Compensator& rhs) = default;

Compensator& Compensator::operator=(const Compensator& rhs)
{
  // self assignment
  if (this == &rhs)
    return *this;

  // base class assignment
  FairModule::operator=(rhs);

  return *this;
}

void Compensator::createMaterials()
{
  auto& matmgr = o2::base::MaterialManager::Instance();

  //
  // Create Materials for Magnetic Compensator
  //
  Int_t isxfld1 = 2.;
  Float_t sxmgmx = 10.;
  o2::base::Detector::initFieldTrackingParams(isxfld1, sxmgmx);

  Int_t isxfld2 = 2; // TODO: set this properly ((AliMagF*)TGeoGlobalMagField::Instance()->GetField())->PrecInteg();

  // ****************
  //     Defines tracking media parameters.
  //     Les valeurs sont commentees pour laisser le defaut
  //     a GEANT (version 3-21, page CONS200), f.m.
  Float_t epsil, stmin, deemax, tmaxfd, stemax;
  epsil = .001;  // Tracking precision,
  stemax = -1.;  // Maximum displacement for multiple scat
  tmaxfd = -20.; // Maximum angle due to field deflection
  deemax = -.3;  // Maximum fractional energy loss, DLS
  stmin = -.8;
  // ***************

  // --- Define the various materials + tracking media for GEANT ---
  //     Aluminum
  matmgr.Material("COMP", 9, "ALUMINIUM0", 26.98, 13., 2.7, 8.9, 37.2);
  matmgr.Medium("COMP", 9, "ALU_C0", 9, 0, isxfld1, sxmgmx, tmaxfd, stemax, deemax, epsil, stmin);
  matmgr.Material("COMP", 29, "ALUMINIUM1", 26.98, 13., 2.7, 8.9, 37.2);
  matmgr.Medium("COMP", 29, "ALU_C1", 29, 0, isxfld1, sxmgmx, tmaxfd, stemax, deemax, epsil, stmin);
  matmgr.Material("COMP", 49, "ALUMINIUM2", 26.98, 13., 2.7, 8.9, 37.2);
  matmgr.Medium("COMP", 49, "ALU_C2", 49, 0, isxfld1, sxmgmx, tmaxfd, stemax, deemax, epsil, stmin);

  //    Iron
  matmgr.Material("COMP", 10, "IRON0", 55.85, 26., 7.87, 1.76, 17.1);
  matmgr.Medium("COMP", 10, "FE_C0", 10, 0, isxfld1, sxmgmx, tmaxfd, stemax, deemax, epsil, stmin);
  matmgr.Material("COMP", 30, "IRON1", 55.85, 26., 7.87, 1.76, 17.1);
  matmgr.Medium("COMP", 30, "FE_C1", 30, 0, 1, sxmgmx, tmaxfd, stemax, deemax, epsil, stmin);
  matmgr.Material("COMP", 50, "IRON2", 55.85, 26., 7.87, 1.76, 17.1);
  matmgr.Medium("COMP", 50, "FE_C2", 50, 0, isxfld1, sxmgmx, tmaxfd, stemax, deemax, epsil, stmin);

  //    Copper
  matmgr.Material("COMP", 17, "COPPER0", 63.55, 29., 8.96, 1.43, 15.1);
  matmgr.Material("COMP", 37, "COPPER1", 63.55, 29., 8.96, 1.43, 15.1);
  matmgr.Material("COMP", 57, "COPPER2", 63.55, 29., 8.96, 1.43, 15.1);
  matmgr.Medium("COMP", 17, "Cu_C0", 17, 0, isxfld1, sxmgmx, tmaxfd, stemax, deemax, epsil, stmin);
  matmgr.Medium("COMP", 37, "Cu_C1", 37, 0, isxfld1, sxmgmx, tmaxfd, stemax, deemax, epsil, stmin);
  matmgr.Medium("COMP", 57, "Cu_C2", 57, 0, isxfld1, sxmgmx, tmaxfd, stemax, deemax, epsil, stmin);
}

void Compensator::ConstructGeometry()
{
  createMaterials();
  createCompensator();
}

#define kDegrad TMath::DegToRad()

void Compensator::createCompensator()
{
  auto top = gGeoManager->GetVolume("cave");
  top->AddNode(createMagnetYoke(), 1, new TGeoTranslation(0., 0., 1075.));
}

void Compensator::SetSpecialPhysicsCuts()
{
  auto& param = o2::passive::HallSimParam::Instance();
  if (!param.fastYoke) {
    return;
  }

  auto& matmgr = o2::base::MaterialManager::Instance();
  using namespace o2::base; // to have enum values of EProc and ECut available

  // NOTE: This is a test setting trying to disable physics
  // processes in a tracking medium; Work in progress

  // need to call method from matmgr
  // 30 == COMP_IRON_C1
  // clang-format off
  matmgr.SpecialProcesses("COMP", 30, {{EProc::kPAIR, 0},
                                       {EProc::kCOMP, 0},
                                       {EProc::kPHOT, 0},
                                       {EProc::kPFIS, 0},
                                       {EProc::kDRAY, 0},
                                       {EProc::kANNI, 0},
                                       {EProc::kBREM, 0},
                                       {EProc::kHADR, 0},
                                       {EProc::kMUNU, 0},
                                       {EProc::kDCAY, 0},
                                       {EProc::kLOSS, 0},
                                       {EProc::kMULS, 0},
                                       {EProc::kRAYL, 0},
                                       {EProc::kLABS, 0}});
  // clang-format on

  // cut settings for the magnet yoke (fast) medium
  const double cut1 = 1;
  const double cutTofmax = 1e10;
  // clang-format off
  matmgr.SpecialCuts("COMP", 30, {{ECut::kCUTGAM, cut1},
                                 {ECut::kCUTELE, cut1},
                                 {ECut::kCUTNEU, cut1},
                                 {ECut::kCUTHAD, cut1},
                                 {ECut::kCUTMUO, cut1},
                                 {ECut::kBCUTE, cut1},
                                 {ECut::kBCUTM, cut1},
                                 {ECut::kDCUTE, cut1},
                                 {ECut::kDCUTM, cut1},
                                 {ECut::kPPCUTM, cut1},
                                 {ECut::kTOFMAX, cutTofmax}});
  // clang-format on
}

TGeoVolume* Compensator::createMagnetYoke()
{
  TGeoVolumeAssembly* voMagnet = new TGeoVolumeAssembly("DCM0");
  voMagnet->SetName("DCM0");
  TGeoRotation* Ry180 = new TGeoRotation("Ry180", 180., 180., 0.);
  auto& matmgr = o2::base::MaterialManager::Instance();
  auto kMedAlu = matmgr.getTGeoMedium("COMP_ALU_C0");
  auto kMedCooper = matmgr.getTGeoMedium("COMP_Cu_C0");
  auto kMedIron = matmgr.getTGeoMedium("COMP_FE_C0");

  // we use a special optimized tracking medium for the inner part
  // of the YOKE (FE_C1 instead of FE_C2)
  auto kMedIronInner = matmgr.getTGeoMedium("COMP_FE_C1");

  const double innerUpLx = 8.;
  const double innerUpLy = 32.2;
  const double innerDwLx = 46.;
  const double innerDwLy = 23.;
  const double outerLx = 116.4;
  const double outerLy = 90.2;
  const double Lz = 250.;

  new TGeoBBox("shMagnetYokeOuter", outerLx / 2.0, outerLy / 2.0, Lz / 2.0);
  new TGeoBBox("shMagnetYokeInnerUp", innerUpLx / 2.0, innerUpLy / 2.0, Lz / 1.0);
  new TGeoBBox("shMagnetYokeInnerDw", innerDwLx / 2.0, innerDwLy / 2.0, Lz / 1.0);
  (new TGeoTranslation("trMagnetYokeOuter", 0.0, -29.1, 0.0))->RegisterYourself();
  (new TGeoTranslation("trMagnetYokeInnerUp", 0.0, 0.0, 0.0))->RegisterYourself();
  (new TGeoTranslation("trMagnetYokeInnerDw", 0.0, -27.5, 0.0))->RegisterYourself();

  TGeoCompositeShape* shMagnetYoke =
    new TGeoCompositeShape("shMagnetBulk",
                           "shMagnetYokeOuter:trMagnetYokeOuter-(shMagnetYokeInnerUp:trMagnetYokeInnerUp+"
                           "shMagnetYokeInnerDw:trMagnetYokeInnerDw)");
  TGeoVolume* voMagnetYoke = new TGeoVolume("voMagnetYoke", shMagnetYoke, kMedIron);

  // make a second version of volume which is smaller than the first and which can be embedded
  // into the first for the purpose of defining a "fast physics" region
  // introduce a thin layer dimension "delta" in which we treat physics correctly
  // and which we can tune
  auto& param = o2::passive::HallSimParam::Instance();
  const double delta = param.yokeDelta;

  new TGeoBBox("shMagnetYokeOuterFast", (outerLx - delta) / 2.0, (outerLy - delta) / 2.0, (Lz - delta) / 2.0);
  new TGeoBBox("shMagnetYokeInnerUpFast", (innerUpLx + delta) / 2.0, (innerUpLy + delta) / 2.0, Lz / 1.0);
  new TGeoBBox("shMagnetYokeInnerDwFast", (innerDwLx + delta) / 2.0, (innerDwLy + delta) / 2.0, Lz / 1.0);

  TGeoCompositeShape* shMagnetYokeFast =
    new TGeoCompositeShape("shMagnetInner",
                           "shMagnetYokeOuterFast:trMagnetYokeOuter-(shMagnetYokeInnerUpFast:trMagnetYokeInnerUp+"
                           "shMagnetYokeInnerDwFast:trMagnetYokeInnerDw)");

  TGeoVolume* voMagnetYokeInner = new TGeoVolume("voMagnetYokeInner", shMagnetYokeFast, kMedIronInner);
  if (delta >= 0.) {
    voMagnetYoke->AddNode(voMagnetYokeInner, 1, new TGeoTranslation(0., 0., 0.0));
  }

  // Make the coils:
  TGeoVolume* voCoilH = gGeoManager->MakeBox("voCoilH", kMedCooper, 12.64 / 2.0, 21.46 / 2.0, 310.5 / 2.0);
  TGeoVolume* voCoilV = gGeoManager->MakeBox("voCoilV", kMedCooper, 12.64 / 2.0, 35.80 / 2.0, 26.9 / 2.0);

  // Make the top coil supports:
  // Polygone Coordinates (x,y)
  Double_t x, y;
  const Double_t kDegToRad = TMath::Pi() / 180.;
  const Double_t AngleInner = 4.5 * kDegToRad;
  const Double_t AngleOuter = 56.0 * kDegToRad;
  const Double_t ArcStart = 90. - AngleOuter / kDegToRad;
  const Double_t ArcEnd = 90. + AngleInner / kDegToRad;
  const Double_t b = 13.6;
  const Double_t Lx = 37.2;
  const Double_t Ly = 25.7;
  const Double_t LxV = 14.9;
  const Double_t R = 9.50;
  const Double_t dz = 2.00 / 2.0;
  const Int_t npoints = 8;
  Double_t CenterX;
  Double_t CenterY;
  Double_t PointsX[npoints] = {0.};
  Double_t PointsY[npoints] = {0.};
  Int_t ip = 0;
  // Start point:
  x = 0.0;
  y = 0.0;
  PointsX[ip] = x;
  PointsY[ip] = y;
  ip++;
  // 1st step:
  x = 0.00;
  y = 1.95;
  PointsX[ip] = x;
  PointsY[ip] = y;
  ip++;
  // 2nd step:
  x += b;
  y += b * TMath::Tan(AngleInner);
  PointsX[ip] = x;
  PointsY[ip] = y;
  ip++;
  // Center of Arc:
  x += R * TMath::Sin(AngleInner);
  y -= R * TMath::Cos(AngleInner);
  CenterX = x;
  CenterY = y;
  TGeoTubeSeg* shPolygonArc = new TGeoTubeSeg("shPolygonArc", R - 2.0, R, dz, ArcStart, ArcEnd);
  (new TGeoTranslation("trPolygonArc", x, y, 0.))->RegisterYourself();
  // 3rd Step:
  x += R * TMath::Sin(AngleOuter);
  y += R * TMath::Cos(AngleOuter);
  PointsX[ip] = x;
  PointsY[ip] = y;
  ip++;
  // 4th Step:
  Double_t a = Lx - b - R * TMath::Sin(AngleInner) - R * TMath::Sin(AngleOuter);
  x = Lx;
  y -= a * TMath::Tan(AngleOuter);
  PointsX[ip] = x;
  PointsY[ip] = y;
  ip++;
  // 5th Step:
  x = Lx;
  y = -Ly;
  PointsX[ip] = x;
  PointsY[ip] = y;
  ip++;
  // 6th Step:
  x = LxV;
  y = -Ly;
  PointsX[ip] = x;
  PointsY[ip] = y;
  ip++;
  // 7th Step:
  x = LxV;
  y = 0.0;
  PointsX[ip] = x;
  PointsY[ip] = y;
  ip++;
  //
  //
  //
  TGeoXtru* shPolygon = new TGeoXtru(2);
  shPolygon->SetNameTitle("shPolygon", "shPolygon");
  shPolygon->DefinePolygon(npoints, PointsX, PointsY);
  shPolygon->DefineSection(0, -dz, 0., 0., 1.0); // index, Z position, offset (x,y) and scale for first section
  shPolygon->DefineSection(1, +dz, 0., 0., 1.0); // idem, second section

  TGeoCompositeShape* shCoilSupportV = new TGeoCompositeShape("shCoilSupportV", "shPolygon+shPolygonArc:trPolygonArc");
  TGeoVolume* voCoilSupportV = new TGeoVolume("voCoilSupportV", shCoilSupportV, kMedAlu);

  const Double_t MagCoilDx = 12.64 / 2.;
  const Double_t MagCoilDy = 21.46 / 2.;
  const Double_t SqOuterDx = MagCoilDx + 2.8;
  const Double_t SqInnerDx = MagCoilDx + 0.6;
  const Double_t SqOuterDy = 29.2 / 2.;
  const Double_t SqInnerDy = 24.8 / 2.;
  const Double_t SqOuterDz = 15.5 / 2.;
  const Double_t SqInnerDz = SqOuterDz * 2.;
  TGeoBBox* shCoilSupportSqOuter = new TGeoBBox("shCoilSupportSqOuter", SqOuterDx, SqOuterDy, SqOuterDz);
  TGeoBBox* shCoilSupportSqInner = new TGeoBBox("shCoilSupportSqInner", SqInnerDx, SqInnerDy, SqInnerDz);
  TGeoCompositeShape* shCoilSupportSq =
    new TGeoCompositeShape("shCoilSupportSq", "shCoilSupportSqOuter - shCoilSupportSqInner");
  TGeoVolume* voCoilSupportSq = new TGeoVolume("voCoilSupportSq", shCoilSupportSq, kMedAlu);

  const Double_t HSuppDx = (Lx - LxV + 0.6) / 2.0;
  const Double_t HSuppDy = 2.2 / 2.0;
  const Double_t HSuppDz = SqOuterDz;

  TGeoVolume* voCoilSupportH = gGeoManager->MakeBox("voCoilSupportH", kMedAlu, HSuppDx, HSuppDy, HSuppDz);

  TGeoVolumeAssembly* voCoilSupport = new TGeoVolumeAssembly("voCoilSupport");
  voCoilSupportV->SetLineColor(kViolet + 9);
  voCoilSupportSq->SetLineColor(kBlue - 5);
  voCoilSupportH->SetLineColor(kPink);
  // voCoilSupportH  -> SetTransparency(16);
  voCoilSupport->AddNode(voCoilSupportV, 1, new TGeoTranslation(SqOuterDx - LxV, SqOuterDy, 0.));
  voCoilSupport->AddNode(voCoilSupportSq, 1, new TGeoTranslation(0., 0., 0.));
  voCoilSupport->AddNode(voCoilSupportH, 1, new TGeoTranslation(SqOuterDx + HSuppDx, SqOuterDy - Ly - HSuppDy, 0.));

  // Make the Top Support for Geodesic reference points:
  TGeoVolume* voSupportHTop = gGeoManager->MakeBox("voSupportHTop", kMedAlu, 66.0 / 2.0, 2.0 / 2.0, 17.0 / 2.0);
  TGeoVolume* voSupportHBot = gGeoManager->MakeBox("voSupportHBot", kMedAlu, 14.0 / 2.0, 2.0 / 2.0, 17.0 / 2.0);
  TGeoVolume* voSupportVert = gGeoManager->MakeBox("voSupportVert", kMedAlu, 3.0 / 2.0, 25.0 / 2.0, 17.0 / 2.0);

  TGeoVolumeAssembly* voSupportGeoRefPoint = new TGeoVolumeAssembly("voSupportGeoRefPoint");
  voSupportHTop->SetLineColor(kGreen);
  voSupportHBot->SetLineColor(kGreen);
  voSupportVert->SetLineColor(kGreen);
  voSupportGeoRefPoint->AddNode(voSupportHTop, 1, new TGeoTranslation(0.0, 28.0, 0.));
  voSupportGeoRefPoint->AddNode(voSupportHBot, 1, new TGeoTranslation(+33.0, 1.0, 0.));
  voSupportGeoRefPoint->AddNode(voSupportHBot, 2, new TGeoTranslation(-33.0, 1.0, 0.));
  voSupportGeoRefPoint->AddNode(voSupportVert, 1, new TGeoTranslation(+31.5, 14.5, 0.));
  voSupportGeoRefPoint->AddNode(voSupportVert, 2, new TGeoTranslation(-31.5, 14.5, 0.));

  // Add some color:
  voMagnetYoke->SetLineColor(kAzure - 7);
  voCoilH->SetLineColor(kOrange - 3);
  voCoilV->SetLineColor(kOrange - 3);
  // Assembling:

  voMagnet->AddNode(voMagnetYoke, 1, new TGeoTranslation(0., 0., 0.0));
  voMagnet->AddNode(voCoilH, 1, new TGeoTranslation(+16.14, +29.83, 0.0));
  voMagnet->AddNode(voCoilH, 2, new TGeoTranslation(-16.14, +29.83, 0.0));
  voMagnet->AddNode(voCoilH, 3, new TGeoTranslation(+16.14, -27.43, 0.0));
  voMagnet->AddNode(voCoilH, 4, new TGeoTranslation(-16.14, -27.43, 0.0));
  voMagnet->AddNode(voCoilV, 1, new TGeoTranslation(+16.14, 1.20, +141.8));
  voMagnet->AddNode(voCoilV, 2, new TGeoTranslation(-16.14, 1.20, +141.8));
  voMagnet->AddNode(voCoilV, 3, new TGeoTranslation(+16.14, 1.20, -141.8));
  voMagnet->AddNode(voCoilV, 4, new TGeoTranslation(-16.14, 1.20, -141.8));
  Double_t zGeoRef = 74.0 / 2. + SqOuterDz + 9.0 + 17.0 / 2.0;
  voMagnet->AddNode(voSupportGeoRefPoint, 1, new TGeoTranslation(0., 16.0, +zGeoRef));
  voMagnet->AddNode(voSupportGeoRefPoint, 2, new TGeoTranslation(0., 16.0, -zGeoRef));
  Double_t zCoilSupp = 29.83 - MagCoilDy - 0.6 + SqInnerDy;
  voMagnet->AddNode(voCoilSupport, 1, new TGeoTranslation(+16.14, zCoilSupp, 74.0 * 0.5));
  voMagnet->AddNode(voCoilSupport, 2, new TGeoTranslation(+16.14, zCoilSupp, -74.0 * 0.5));
  voMagnet->AddNode(voCoilSupport, 3, new TGeoTranslation(+16.14, zCoilSupp, 74.0 * 1.5));
  voMagnet->AddNode(voCoilSupport, 4, new TGeoTranslation(+16.14, zCoilSupp, -74.0 * 1.5));
  //
  voMagnet->AddNode(voCoilSupport, 5, new TGeoCombiTrans(-16.14, zCoilSupp, 74.0 * 0.5, Ry180));
  voMagnet->AddNode(voCoilSupport, 6, new TGeoCombiTrans(-16.14, zCoilSupp, -74.0 * 0.5, Ry180));
  voMagnet->AddNode(voCoilSupport, 7, new TGeoCombiTrans(-16.14, zCoilSupp, 74.0 * 1.5, Ry180));
  voMagnet->AddNode(voCoilSupport, 8, new TGeoCombiTrans(-16.14, zCoilSupp, -74.0 * 1.5, Ry180));

  return (TGeoVolume*)voMagnet;
}

FairModule* Compensator::CloneModule() const { return new Compensator(*this); }
ClassImp(o2::passive::Compensator);
