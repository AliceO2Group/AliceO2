// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "Field/MagneticField.h"
#include "DetectorsPassive/FrameStructure.h"
#include "DetectorsBase/Detector.h"
#include "DetectorsBase/MaterialManager.h"
#include <TGeoArb8.h>
#include <TGeoBBox.h>
#include <TGeoBoolNode.h>
#include <TGeoCompositeShape.h>
#include <TGeoManager.h>
#include <TGeoMatrix.h>
#include <TGeoPgon.h>
#include <TGeoTrd1.h>
#include <TGeoTube.h>
#include <TGeoXtru.h>
#include <TMath.h>
#include <TString.h>
#include <TVirtualMC.h>

#include <cassert>

ClassImp(o2::passive::FrameStructure);

namespace o2
{
namespace passive
{
const char* TOPNAME = "cave";
#define AliMatrix TVirtualMC::GetMC()->Matrix
#define kRaddeg TMath::RadToDeg()
#define kDegrad TMath::DegToRad()

FrameStructure::FrameStructure(const char* name, const char* Title) : FairModule(name, Title)
{
}

FrameStructure::FrameStructure(const FrameStructure& rhs) : FairModule(rhs)
{
}

FairModule* FrameStructure::CloneModule() const
{
  std::cout << "cloning FrameStructure - now the right one !" << std::endl;
  return new FrameStructure(*this);
}

void FrameStructure::makeHeatScreen(const char* name, float dyP, int rot1, int rot2)
{
  // Heat screen panel
  //
  auto vmc = TVirtualMC::GetMC();
  const int kAir = mAirMedID;
  const int kAlu = mAluMedID;

  float dx, dy;
  char mname[16];
  char cname[16];
  char t1name[16];
  char t2name[16];
  char t3name[16];
  char t4name[16];
  char t5name[16];

  //
  float dxP = 2. * (287. * TMath::Sin(10. * TMath::Pi() / 180.) - 2.);
  float dzP = 1.05;
  //
  // Mother volume
  float thshM[3];
  thshM[0] = dxP / 2.;
  thshM[1] = dyP / 2.;
  thshM[2] = dzP / 2.;
  snprintf(mname, 16, "BTSH_%s", name);
  vmc->Gsvolu(mname, "BOX ", kAir, thshM, 3);
  //
  // Aluminum sheet
  thshM[2] = 0.025;
  snprintf(cname, 16, "BTSHA_%s", name);
  vmc->Gsvolu(cname, "BOX ", kAlu, thshM, 3);
  vmc->Gspos(cname, 1, mname, 0., 0., -0.5, 0);
  //
  // Tubes
  float thshT[3];
  thshT[0] = 0.4;
  thshT[1] = 0.5;
  thshT[2] = (dyP / 2. - 8.);
  //
  snprintf(t1name, 16, "BTSHT1_%s", name);
  vmc->Gsvolu(t1name, "TUBE", kAlu, thshT, 3);
  dx = -dxP / 2. + 8. - 0.5;
  vmc->Gspos(t1name, 1, mname, dx, 0., 0.025, rot1);
  //
  snprintf(t2name, 16, "BTSHT2_%s", name);
  snprintf(t3name, 16, "BTSHT3_%s", name);
  snprintf(t4name, 16, "BTSHT4_%s", name);
  snprintf(t5name, 16, "BTSHT5_%s", name);
  thshT[2] = (thshM[1] - 12.);
  vmc->Gsvolu(t2name, "TUBE", kAlu, thshT, 3);
  thshT[2] = 7.9 / 2.;
  vmc->Gsvolu(t3name, "TUBE", kAlu, thshT, 3);
  thshT[2] = 23.9 / 2.;
  vmc->Gsvolu(t4name, "TUBE", kAlu, thshT, 3);

  int sig = 1;
  int ipo = 1;
  for (int i = 0; i < 5; i++) {
    sig *= -1;
    dx += 8.00;
    dy = 4. * sig;
    float dy1 = -(thshM[1] - 15.5) * sig;
    float dy2 = -(thshM[1] - 7.5) * sig;

    vmc->Gspos(t2name, ipo++, mname, dx, dy, 0.025, rot1);
    dx += 6.9;
    vmc->Gspos(t2name, ipo++, mname, dx, dy, 0.025, rot1);

    vmc->Gspos(t3name, i + 1, mname, dx - 3.45, dy1, 0.025, rot2);
    vmc->Gspos(t4name, i + 1, mname, dx - 3.45, dy2, 0.025, rot2);
  }
  dx += 8.;
  vmc->Gspos(t1name, 2, mname, dx, 0., 0.025, rot1);
  vmc->Gspos(t3name, 6, mname, dx - 3.45, -(thshM[1] - 7.5), 0.025, rot2);
}

void FrameStructure::createWebFrame(const char* name, float dHz, float theta0, float phi0)
{
  //
  // Create a web frame element
  //
  auto vmc = TVirtualMC::GetMC();

  phi0 = 0.;
  const float krad2deg = 180. / TMath::Pi();
  const float kdeg2rad = 1. / krad2deg;
  const int kAir = mAirMedID;
  const int kSteel = mSteelMedID;

  float ptrap[11];
  char nameA[16];
  snprintf(nameA, 16, "%sA", name);

  char nameI[16];
  snprintf(nameI, 16, "%sI", name);

  theta0 *= kdeg2rad;
  phi0 *= kdeg2rad;
  float theta = TMath::Pi() / 2.;
  float phi = TMath::ACos(TMath::Cos(theta0) * TMath::Cos(phi0));

  if (phi0 < 0)
    phi = -phi;

  phi *= krad2deg;
  theta *= krad2deg;

  ptrap[0] = dHz / 2;
  ptrap[2] = theta;
  ptrap[1] = phi;
  ptrap[3] = 6. / cos(theta0 * kdeg2rad) / 2.;
  ptrap[4] = 1.;
  ptrap[5] = ptrap[4];
  ptrap[6] = 0;
  ptrap[7] = ptrap[3];
  ptrap[8] = ptrap[4];
  ptrap[9] = ptrap[4];
  ptrap[10] = 0;
  vmc->Gsvolu(name, "TRAP", kSteel, ptrap, 11);
  vmc->Gsvolu(nameI, "TRAP", kSteel, ptrap, 11);
  ptrap[3] = (6. - 1.) / cos(theta0 * kdeg2rad) / 2.;
  ptrap[4] = 0.75;
  ptrap[5] = ptrap[4];
  ptrap[7] = ptrap[3];
  ptrap[8] = ptrap[4];
  ptrap[9] = ptrap[4];

  vmc->Gsvolu(nameA, "TRAP", kAir, ptrap, 11);
  vmc->Gspos(nameA, 1, name, -0.25, 0.0, 0., 0, "ONLY");
  vmc->Gspos(nameA, 2, nameI, +0.25, 0.0, 0., 0, "ONLY");
  gGeoManager->GetVolume(name)->SetVisContainers();
  gGeoManager->GetVolume(nameI)->SetVisContainers();
}

TGeoCompositeShape* FrameStructure::createTOFRail(float y)
{
  char nameSostA1[16];
  snprintf(nameSostA1, 16, "SostA1");
  TGeoBBox* boxSostA1 = new TGeoBBox(nameSostA1, 0.5, y, 2.0);
  char nameCV[16];
  snprintf(nameCV, 16, "CV");
  TGeoArb8* CV = new TGeoArb8(nameCV, 2.35);
  CV->SetVertex(0, 0.09, -y);
  CV->SetVertex(1, 0.09, y);
  CV->SetVertex(2, 0.89, y);
  CV->SetVertex(3, 0.89, -y);
  CV->SetVertex(4, -0.89, -y);
  CV->SetVertex(5, -0.89, y);
  CV->SetVertex(6, -0.09, y);
  CV->SetVertex(7, -0.09, -y);

  char nameCOB[16];
  snprintf(nameCOB, 16, "COB");
  TGeoBBox* boxCOB = new TGeoBBox(nameCOB, 2.0, y, 0.4);
  char nameCOT[16];
  snprintf(nameCOT, 16, "COT");
  TGeoBBox* boxCOT = new TGeoBBox(nameCOT, 1.7, y, 0.4);

  TGeoTranslation* trCOB = new TGeoTranslation("trCOB", 2.09, 0., -2.75);
  trCOB->RegisterYourself();
  TGeoTranslation* trCOT = new TGeoTranslation("trCOT", 0.81, 0., +2.75);
  trCOT->RegisterYourself();
  TGeoTranslation* trSostA1 = new TGeoTranslation("trSostA1", 2.39, 0., -0.35);
  trSostA1->RegisterYourself();

  TGeoCompositeShape* btofS1 = new TGeoCompositeShape("BtofS1", "CV+(COB:trCOB)+(COT:trCOT)+(SostA1:trSostA1)");
  return btofS1;
}

void FrameStructure::createMaterials()
{
  auto& matmgr = o2::base::MaterialManager::Instance();
  // Creates the materials
  float epsil, stemax, tmaxfd, deemax, stmin;

  epsil = 1.e-4;  // Tracking precision,
  stemax = -0.01; // Maximum displacement for multiple scat
  tmaxfd = -20.;  // Maximum angle due to field deflection
  deemax = -.3;   // Maximum fractional energy loss, DLS
  stmin = -.8;
  int isxfld = 2;     // field uniformity value as defined by Geant3
  float sxmgmx = 10.; // max field
  o2::base::Detector::initFieldTrackingParams(isxfld, sxmgmx);

  float asteel[4] = {55.847, 51.9961, 58.6934, 28.0855};
  float zsteel[4] = {26., 24., 28., 14.};
  float wsteel[4] = {.715, .18, .1, .005};

  // Air

  float aAir[4] = {12.0107, 14.0067, 15.9994, 39.948};
  float zAir[4] = {6., 7., 8., 18.};
  float wAir[4] = {0.000124, 0.755267, 0.231781, 0.012827};
  float dAir = 1.20479E-3;

  // G10
  // G10 60% SiO2 40% epoxy
  float ag10[4] = {12.01, 1., 15.994, 28.086};
  float zg10[4] = {6., 1., 8., 14.};
  float wg10[4] = {0.194, 0.023, 0.443, 0.340};

  matmgr.Mixture("FRAME", 22, "G10", ag10, zg10, 1.7, 4, wg10);
  matmgr.Mixture("FRAME", 65, "STEEL$", asteel, zsteel, 7.88, 4, wsteel);
  matmgr.Mixture("FRAME", 5, "AIR$      ", aAir, zAir, dAir, 4, wAir);
  matmgr.Material("FRAME", 9, "ALU      ", 26.98, 13., 2.7, 8.9, 37.2);

  matmgr.Medium("FRAME", 65, "Steel", 65, 0, isxfld, sxmgmx, tmaxfd, stemax, deemax, epsil, stmin);
  matmgr.Medium("FRAME", 5, "Air", 5, 0, isxfld, sxmgmx, tmaxfd, stemax, deemax, epsil, stmin);
  matmgr.Medium("FRAME", 9, "Aluminum", 9, 0, isxfld, sxmgmx, tmaxfd, stemax, deemax, epsil, stmin);
  matmgr.Medium("FRAME", 22, "G10", 22, 0, isxfld, sxmgmx, tmaxfd, stemax, deemax, epsil, stmin);

  mAirMedID = matmgr.getMediumID("FRAME", 5);
  mSteelMedID = matmgr.getMediumID("FRAME", 65);
  mAluMedID = matmgr.getMediumID("FRAME", 9);
  mG10MedID = matmgr.getMediumID("FRAME", 22);
  // do a cross check
  assert(gGeoManager->GetMedium("FRAME_Air")->GetId() == mAirMedID);
  assert(gGeoManager->GetMedium("FRAME_Aluminum")->GetId() == mAluMedID);
  assert(gGeoManager->GetMedium("FRAME_Steel")->GetId() == mSteelMedID);
  assert(gGeoManager->GetMedium("FRAME_G10")->GetId() == mG10MedID);
}

void FrameStructure::ConstructGeometry()
{
  auto vmc = TVirtualMC::GetMC();

  // TODO: we should be allowed to call this function only once
  if (vmc == nullptr) {
    throw std::runtime_error("VMC instance not initialized");
  }

  // verify that we have the world volume already setup
  if (gGeoManager != nullptr && gGeoManager->GetVolume(TOPNAME)) {
    mCaveIsAvailable = true;
  }

  // create materials
  createMaterials();

  int idrotm[2299];

  AliMatrix(idrotm[2070], 90.0, 0.0, 90.0, 270.0, 0.0, 0.0);
  //
  AliMatrix(idrotm[2083], 170.0, 0.0, 90.0, 90.0, 80.0, 0.0);
  AliMatrix(idrotm[2084], 170.0, 180.0, 90.0, 90.0, 80.0, 180.0);
  AliMatrix(idrotm[2085], 90.0, 180.0, 90.0, 90.0, 0.0, 0.0);
  //
  AliMatrix(idrotm[2086], 80.0, 0.0, 90.0, 90., -10.0, 0.0);
  AliMatrix(idrotm[2096], 100.0, 0.0, 90.0, 90., 10.0, 0.0);
  //
  AliMatrix(idrotm[2087], -100.0, 0.0, 90.0, 270., -10.0, 0.0);
  AliMatrix(idrotm[2097], -80.0, 0.0, 90.0, 270., 10.0, 0.0);

  //
  AliMatrix(idrotm[2088], 90.0, 180.0, 90.0, 270., 0.0, 0.0);
  AliMatrix(idrotm[2089], 90.0, -90.0, 90.0, 0., 0.0, 0.0);
  //
  AliMatrix(idrotm[2090], 90.0, 0.0, 0.0, 0., 90.0, 90.0);
  AliMatrix(idrotm[2091], 0.0, 0.0, 90.0, 90., 90.0, 0.0);
  //
  // Matrices have been imported from Euclid. Some simplification
  // seems possible
  //

  AliMatrix(idrotm[2003], 0.0, 0.0, 90.0, 130.0, 90.0, 40.0);
  AliMatrix(idrotm[2004], 180.0, 0.0, 90.0, 130.0, 90.0, 40.0);
  AliMatrix(idrotm[2005], 180.0, 0.0, 90.0, 150.0, 90.0, 240.0);
  AliMatrix(idrotm[2006], 0.0, 0.0, 90.0, 150.0, 90.0, 240.0);
  AliMatrix(idrotm[2007], 0.0, 0.0, 90.0, 170.0, 90.0, 80.0);
  AliMatrix(idrotm[2008], 180.0, 0.0, 90.0, 190.0, 90.0, 280.0);
  AliMatrix(idrotm[2009], 180.0, 0.0, 90.0, 170.0, 90.0, 80.0);
  AliMatrix(idrotm[2010], 0.0, 0.0, 90.0, 190.0, 90.0, 280.0);
  AliMatrix(idrotm[2011], 0.0, 0.0, 90.0, 350.0, 90.0, 260.0);
  AliMatrix(idrotm[2012], 180.0, 0.0, 90.0, 350.0, 90.0, 260.0);
  AliMatrix(idrotm[2013], 180.0, 0.0, 90.0, 10.0, 90.0, 100.0);
  AliMatrix(idrotm[2014], 0.0, 0.0, 90.0, 10.0, 90.0, 100.0);
  AliMatrix(idrotm[2015], 0.0, 0.0, 90.0, 30.0, 90.0, 300.0);
  AliMatrix(idrotm[2016], 180.0, 0.0, 90.0, 30.0, 90.0, 300.0);
  AliMatrix(idrotm[2017], 180.0, 0.0, 90.0, 50.0, 90.0, 140.0);
  AliMatrix(idrotm[2018], 0.0, 0.0, 90.0, 50.0, 90.0, 140.0);

  AliMatrix(idrotm[2019], 180.0, 0.0, 90.0, 130.0, 90.0, 220.0);
  AliMatrix(idrotm[2020], 180.0, 0.0, 90.0, 50.0, 90.0, 320.0);
  AliMatrix(idrotm[2021], 180.0, 0.0, 90.0, 150.0, 90.0, 60.0);
  AliMatrix(idrotm[2022], 180.0, 0.0, 90.0, 30.0, 90.0, 120.0);
  AliMatrix(idrotm[2023], 180.0, 0.0, 90.0, 170.0, 90.0, 260.0);
  AliMatrix(idrotm[2024], 180.0, 0.0, 90.0, 190.0, 90.0, 100.0);
  AliMatrix(idrotm[2025], 180.0, 0.0, 90.0, 350.0, 90.0, 80.0);
  AliMatrix(idrotm[2026], 180.0, 0.0, 90.0, 10.0, 90.0, 280.0);

  AliMatrix(idrotm[2100], 180.0, 0.0, 90.0, 210.0, 90.0, 120.0);
  AliMatrix(idrotm[2101], 180.0, 0.0, 90.0, 330.0, 90.0, 60.0);

  AliMatrix(idrotm[2027], 0.0, 0.0, 90.0, 50.0, 90.0, 320.0);
  AliMatrix(idrotm[2028], 0.0, 0.0, 90.0, 150.0, 90.0, 60.0);
  AliMatrix(idrotm[2029], 0.0, 0.0, 90.0, 30.0, 90.0, 120.0);
  AliMatrix(idrotm[2030], 0.0, 0.0, 90.0, 10.0, 90.0, 280.0);
  AliMatrix(idrotm[2031], 0.0, 0.0, 90.0, 170.0, 90.0, 260.0);
  AliMatrix(idrotm[2032], 0.0, 0.0, 90.0, 190.0, 90.0, 100.0);
  AliMatrix(idrotm[2033], 0.0, 0.0, 90.0, 350.0, 90.0, 80.0);

  //
  // The Main Space Frame
  // ALIP2A__0007
  // ALIP2A__0008
  //
  float pbox[3], ptrap[11], ptrd1[4], ppgon[10];
  float dx, dy, dz;
  int i, j;
  int jmod = 0;
  //
  // Constants
  //
  // Materials
  const TGeoMedium* kMedAir = gGeoManager->GetMedium("FRAME_Air");
  const TGeoMedium* kMedSteel = gGeoManager->GetMedium("FRAME_Steel");
  const TGeoMedium* kMedAlu = gGeoManager->GetMedium("FRAME_Aluminum");
  const int kAir = mAirMedID;
  const int kSteel = mSteelMedID;
  const int kAlu = mAluMedID;
  const int kG10 = mG10MedID;
  // Angles
  const float kEps = 0.01;
  const float krad2deg = 180. / TMath::Pi();
  const float kdeg2rad = 1. / krad2deg;
  const float sin10 = TMath::Sin(10. * kdeg2rad);
  const float tan10 = TMath::Tan(10. * kdeg2rad);
  const float cos10 = TMath::Cos(10. * kdeg2rad);
  // Dimensions
  // vertical distance of frame wrt to origin (center of inner rings)
  const float hR = 286.00;
  // Height of inner frame from lower edge to outer ring (sectors for detectors)
  const float iFrH = 119.00;
  //
  // radial length of web frame elements
  const float dHz = 113. / cos10 - 0.3; // 114.74 (114.5 on drawing)
  // Positions of ring bars (ALIP2A_0008)
  // outer
  const float dymodU[3] = {71.5, 228.5, 339.5};
  // inner
  const float dymodL[3] = {50.0, 175.0, 297.5};
  //
  // orientation of web frame elements
  const float dymodO[5] = {10., -40., 20., -27.1, 18.4};
  // Position of web frame elements
  float dymodW[5] = {70., 73.6, 224.5, 231.4, 340.2};
  for (int ii = 0; ii < 5; ii++) {
    dymodW[ii] = dymodW[ii] - 3. * TMath::Tan(dymodO[ii] * kdeg2rad);
  }
  // Inner ring bars (Pos 6)
  const float ringH = 6.00;  // Hight
  const float ringW = 10.00; // Width  of the ring bars in z
  const float ringT = 1.00;  // Thickness of bars
  // inner longitudinal bars 4 x 6
  const float longH = 6.00; // Height
  const float longW = 4.00; // Width
  // outer longitudianl bars 8 x 8
  // const float longOD =   8.0;
  // some extra space for mother volume
  const float dext = sin10 * longW / 2. + 0.01;
  // sector hight with extra space
  const float iFrH0 = iFrH + dext;
  // length of inner longitudinal bars
  // inner
  const float longLI = 615.;
  const float zE = 376.5;
  //
  // Frame mother volume
  //
  TGeoPgon* shB77A = new TGeoPgon(0., 360., 18, 2);
  shB77A->SetName("shB77A");
  shB77A->DefineSection(0, -zE, 280., 423.7);
  shB77A->DefineSection(1, zE, 280., 423.7);
  TGeoBBox* shB77B = new TGeoBBox(3.42, 2., 375.5);
  shB77B->SetName("shB77B");
  TGeoTranslation* trB77A = new TGeoTranslation("trB77A", +283.32, 0., 0.);
  TGeoTranslation* trB77B = new TGeoTranslation("trB77B", -283.32, 0., 0.);
  trB77A->RegisterYourself();
  trB77B->RegisterYourself();
  TGeoCompositeShape* shB77 = new TGeoCompositeShape("shB77", "shB77A+shB77B:trB77A+shB77B:trB77B");
  TGeoVolume* voB77 = new TGeoVolume("B077", shB77, gGeoManager->GetMedium("FRAME_Air"));
  voB77->SetName("B077"); // just to avoid a warning
  vmc->Gspos("B077", 1, TOPNAME, 0., 0., 0., 0, "ONLY");
  //
  // Reference plane #1 for TRD
  TGeoPgon* shBREFA = new TGeoPgon(0.0, 360., 18, 2);
  shBREFA->DefineSection(0, -376., 280., 280.1);
  shBREFA->DefineSection(1, 376., 280., 280.1);
  shBREFA->SetName("shBREFA");
  TGeoCompositeShape* shBREF1 = new TGeoCompositeShape("shBREF1", "shBREFA-(shB77B:trB77A+shB77B:trB77B)");
  TGeoVolume* voBREF = new TGeoVolume("BREF1", shBREF1, gGeoManager->GetMedium("FRAME_Air"));
  voBREF->SetVisibility(0);
  vmc->Gspos("BREF1", 1, "B077", 0., 0., 0., 0, "ONLY");
  //
  //  The outer Frame
  //

  float dol = 4.;
  float doh = 4.;
  float ds = 0.63;
  //
  // Rings
  //
  dz = 2. * 410.2 * sin10 - 2. * dol * cos10 - 2. * doh * tan10;
  float l1 = dz / 2.;
  float l2 = dz / 2. + 2. * doh * tan10;

  TGeoVolumeAssembly* asBI42 = new TGeoVolumeAssembly("BI42");
  // Horizontal
  ptrd1[0] = l2 - 0.6 * tan10;
  ptrd1[1] = l2;
  ptrd1[2] = 8.0 / 2.;
  ptrd1[3] = 0.6 / 2.;
  vmc->Gsvolu("BIH142", "TRD1", kSteel, ptrd1, 4);
  ptrd1[0] = l1;
  ptrd1[1] = l1 + 0.6 * tan10;
  ptrd1[2] = 8.0 / 2.;
  ptrd1[3] = 0.6 / 2.;
  vmc->Gsvolu("BIH242", "TRD1", kSteel, ptrd1, 4);

  // Vertical
  ptrd1[0] = l1 + 0.6 * tan10;
  ptrd1[1] = l2 - 0.6 * tan10;
  ptrd1[2] = 0.8 / 2.;
  ptrd1[3] = 6.8 / 2.;
  vmc->Gsvolu("BIV42", "TRD1", kSteel, ptrd1, 4);
  // Place
  asBI42->AddNode(gGeoManager->GetVolume("BIV42"), 1, new TGeoTranslation(0., 0., 0.0));
  asBI42->AddNode(gGeoManager->GetVolume("BIH142"), 1, new TGeoTranslation(0., 0., 3.7));
  asBI42->AddNode(gGeoManager->GetVolume("BIH242"), 1, new TGeoTranslation(0., 0., -3.7));
  //
  // longitudinal bars
  //
  // 80 x 80 x 6.3
  //
  pbox[0] = dol;
  pbox[1] = doh;
  pbox[2] = 345.;
  vmc->Gsvolu("B033", "BOX", kSteel, pbox, 3);
  pbox[0] = dol - ds;
  pbox[1] = doh - ds;
  vmc->Gsvolu("B034", "BOX", kAir, pbox, 3);
  vmc->Gspos("B034", 1, "B033", 0., 0., 0., 0, "ONLY");

  //
  // TPC support
  //
  pbox[0] = 3.37;
  pbox[1] = 2.0;
  pbox[2] = longLI / 2.;
  vmc->Gsvolu("B080", "BOX", kSteel, pbox, 3);
  pbox[0] = 2.78;
  pbox[1] = 1.40;
  pbox[2] = longLI / 2.;
  vmc->Gsvolu("B081", "BOX", kAir, pbox, 3);
  vmc->Gspos("B081", 1, "B080", 0., 0., 0., 0, "ONLY");

  // Small 2nd reference plane elemenet
  pbox[0] = 0.05;
  pbox[1] = 2.0;
  pbox[2] = longLI / 2.;
  vmc->Gsvolu("BREF2", "BOX", kAir, pbox, 3);
  vmc->Gspos("BREF2", 1, "B080", 3.37 - 0.05, 0., 0., 0, "ONLY");

  vmc->Gspos("B080", 1, "B077", 283.25, 0., 0., 0, "ONLY");
  vmc->Gspos("B080", 2, "B077", -283.25, 0., 0., idrotm[2088], "ONLY");

  //
  // Diagonal bars (1)
  //
  float h, d, dq, x, theta;

  h = (dymodU[1] - dymodU[0] - 2. * dol) * .999;
  d = 2. * dol;
  dq = h * h + dz * dz;

  x = TMath::Sqrt((dz * dz - d * d) / dq + d * d * h * h / dq / dq) + d * h / dq;

  theta = krad2deg * TMath::ACos(x);

  ptrap[0] = dz / 2.;
  ptrap[1] = theta;
  ptrap[2] = 0.;
  ptrap[3] = doh;
  ptrap[4] = dol / x;
  ptrap[5] = ptrap[4];
  ptrap[6] = 0;
  ptrap[7] = ptrap[3];
  ptrap[8] = ptrap[4];
  ptrap[9] = ptrap[4];
  ptrap[10] = 0;

  vmc->Gsvolu("B047", "TRAP", kSteel, ptrap, 11);
  ptrap[3] = doh - ds;
  ptrap[4] = (dol - ds) / x;
  ptrap[5] = ptrap[4];
  ptrap[7] = ptrap[3];
  ptrap[8] = ptrap[4];
  ptrap[9] = ptrap[4];
  vmc->Gsvolu("B048", "TRAP", kAir, ptrap, 11);
  vmc->Gspos("B048", 1, "B047", 0.0, 0.0, 0., 0, "ONLY");

  /*
   Crosses (inner most)
         \\  //
          \\//
          //\\
         //  \\
  */
  h = (2. * dymodU[0] - 2. * dol) * .999;
  //
  // Mother volume
  //
  pbox[0] = h / 2;
  pbox[1] = doh;
  pbox[2] = dz / 2.;
  vmc->Gsvolu("BM49", "BOX ", kAir, pbox, 3);

  dq = h * h + dz * dz;
  x = TMath::Sqrt((dz * dz - d * d) / dq + d * d * h * h / dq / dq) + d * h / dq;
  theta = krad2deg * TMath::ACos(x);

  ptrap[0] = dz / 2. - kEps;
  ptrap[1] = theta;
  ptrap[2] = 0.;
  ptrap[3] = doh - kEps;
  ptrap[4] = dol / x;
  ptrap[5] = ptrap[4];
  ptrap[7] = ptrap[3];
  ptrap[8] = ptrap[4];
  ptrap[9] = ptrap[4];

  vmc->Gsvolu("B049", "TRAP", kSteel, ptrap, 11);
  ptrap[0] = ptrap[0] - kEps;
  ptrap[3] = (doh - ds);
  ptrap[4] = (dol - ds) / x;
  ptrap[5] = ptrap[4];
  ptrap[7] = ptrap[3];
  ptrap[8] = ptrap[4];
  ptrap[9] = ptrap[4];
  vmc->Gsvolu("B050", "TRAP", kAir, ptrap, 11);
  vmc->Gspos("B050", 1, "B049", 0.0, 0.0, 0., 0, "ONLY");
  vmc->Gspos("B049", 1, "BM49", 0.0, 0.0, 0., 0, "ONLY");

  float dd1 = d * TMath::Tan(theta * kdeg2rad);
  float dd2 = d / TMath::Tan(2. * theta * kdeg2rad);
  float theta2 = TMath::ATan(TMath::Abs(dd2 - dd1) / d / 2.);

  ptrap[0] = dol;
  ptrap[1] = theta2 * krad2deg;
  ptrap[2] = 0.;
  ptrap[3] = doh;
  ptrap[4] = (dz / 2. / x - dd1 - dd2) / 2.;
  ptrap[5] = ptrap[4];
  ptrap[6] = 0.;
  ptrap[7] = ptrap[3];
  ptrap[8] = dz / 4. / x;
  ptrap[9] = ptrap[8];

  vmc->Gsvolu("B051", "TRAP", kSteel, ptrap, 11);
  float ddx0 = ptrap[8];

  float dd1s = dd1 * (1. - 2. * ds / d);
  float dd2s = dd2 * (1. - 2. * ds / d);
  float theta2s = TMath::ATan(TMath::Abs(dd2s - dd1s) / (d - 2. * ds) / 2.);

  ptrap[0] = dol - ds;
  ptrap[1] = theta2s * krad2deg;
  ptrap[2] = 0.;
  ptrap[3] = doh - ds;
  ptrap[4] = ptrap[4] + ds / d / 2. * (dd1 + dd2);
  ptrap[5] = ptrap[4];
  ptrap[6] = 0.;
  ptrap[7] = ptrap[3];
  ptrap[8] = ptrap[8] - ds / 2. / d * (dd1 + dd2);
  ptrap[9] = ptrap[8];

  vmc->Gsvolu("B052", "TRAP", kAir, ptrap, 11);
  vmc->Gspos("B052", 1, "B051", 0.0, 0.0, 0., 0, "ONLY");

  float ddx, ddz, drx, drz, rtheta;

  AliMatrix(idrotm[2001], -theta + 180, 0.0, 90.0, 90.0, 90. - theta, 0.0);
  rtheta = (90. - theta) * kdeg2rad;
  ddx = -ddx0 - dol * TMath::Tan(theta2);
  ddz = -dol;

  drx = TMath::Cos(rtheta) * ddx + TMath::Sin(rtheta) * ddz + pbox[0];
  drz = -TMath::Sin(rtheta) * ddx + TMath::Cos(rtheta) * ddz - pbox[2];
  vmc->Gspos("B051", 1, "BM49", drx, 0.0, drz, idrotm[2001], "ONLY");

  AliMatrix(idrotm[2002], -theta, 0.0, 90.0, 90.0, 270. - theta, 0.0);
  rtheta = (270. - theta) * kdeg2rad;

  drx = TMath::Cos(rtheta) * ddx + TMath::Sin(rtheta) * ddz - pbox[0];
  drz = -TMath::Sin(rtheta) * ddx + TMath::Cos(rtheta) * ddz + pbox[2];
  vmc->Gspos("B051", 2, "BM49", drx, 0.0, drz, idrotm[2002], "ONLY");

  //
  // Diagonal bars (3)
  //
  h = ((dymodU[2] - dymodU[1]) - 2. * dol) * .999;
  dq = h * h + dz * dz;
  x = TMath::Sqrt((dz * dz - d * d) / dq + d * d * h * h / dq / dq) + d * h / dq;
  theta = krad2deg * TMath::ACos(x);

  ptrap[0] = dz / 2.;
  ptrap[1] = theta;
  ptrap[3] = doh;
  ptrap[4] = dol / x;
  ptrap[5] = ptrap[4];
  ptrap[7] = ptrap[3];
  ptrap[8] = ptrap[4];
  ptrap[9] = ptrap[4];

  vmc->Gsvolu("B045", "TRAP", kSteel, ptrap, 11);
  ptrap[3] = doh - ds;
  ptrap[4] = (dol - ds) / x;
  ptrap[5] = ptrap[4];
  ptrap[7] = ptrap[3];
  ptrap[8] = ptrap[4];
  ptrap[9] = ptrap[4];
  vmc->Gsvolu("B046", "TRAP", kAir, ptrap, 11);
  vmc->Gspos("B046", 1, "B045", 0.0, 0.0, 0., 0, "ONLY");

  //
  // Positioning of diagonal bars

  float rd = 405.5 + 0.51;
  dz = (dymodU[1] + dymodU[0]) / 2.;
  float dz2 = (dymodU[1] + dymodU[2]) / 2.;

  //
  //  phi = 60
  //

  float phi = 60;
  dx = rd * TMath::Sin(phi * kdeg2rad);
  dy = rd * TMath::Cos(phi * kdeg2rad);

  vmc->Gspos("B045", 1, "B077", -dx, dy, dz2, idrotm[2021], "ONLY");
  vmc->Gspos("B045", 2, "B077", -dx, dy, -dz2, idrotm[2028], "ONLY");
  vmc->Gspos("B045", 3, "B077", dx, dy, dz2, idrotm[2022], "ONLY");
  vmc->Gspos("B045", 4, "B077", dx, dy, -dz2, idrotm[2029], "ONLY");

  vmc->Gspos("B045", 5, "B077", dx, -dy, -dz2, idrotm[2021], "ONLY");
  vmc->Gspos("B045", 6, "B077", dx, -dy, +dz2, idrotm[2028], "ONLY");
  vmc->Gspos("B045", 7, "B077", -dx, -dy, -dz2, idrotm[2022], "ONLY");
  vmc->Gspos("B045", 8, "B077", -dx, -dy, +dz2, idrotm[2029], "ONLY");

  vmc->Gspos("B047", 1, "B077", -dx, -dy, dz, idrotm[2022], "ONLY");
  vmc->Gspos("B047", 2, "B077", -dx, -dy, -dz, idrotm[2029], "ONLY");

  vmc->Gspos("B047", 3, "B077", dx, -dy, dz, idrotm[2021], "ONLY");
  vmc->Gspos("B047", 4, "B077", dx, -dy, -dz, idrotm[2028], "ONLY");

  vmc->Gspos("BM49", 1, "B077", dx, -dy, 0., idrotm[2101], "ONLY");
  vmc->Gspos("BM49", 2, "B077", -dx, -dy, 0., idrotm[2100], "ONLY");

  //
  //  phi = 80
  //

  phi = 80;
  dx = rd * TMath::Sin(phi * kdeg2rad);
  dy = rd * TMath::Cos(phi * kdeg2rad);

  vmc->Gspos("B047", 13, "B077", -dx, -dy, dz, idrotm[2008], "ONLY");
  vmc->Gspos("B047", 14, "B077", -dx, -dy, -dz, idrotm[2010], "ONLY");
  vmc->Gspos("B047", 15, "B077", dx, -dy, dz, idrotm[2012], "ONLY");
  vmc->Gspos("B047", 16, "B077", dx, -dy, -dz, idrotm[2011], "ONLY");

  vmc->Gspos("B045", 11, "B077", -dx, dy, dz2, idrotm[2023], "ONLY");
  vmc->Gspos("B045", 12, "B077", -dx, dy, -dz2, idrotm[2031], "ONLY");
  vmc->Gspos("B045", 13, "B077", dx, dy, dz2, idrotm[2026], "ONLY");
  vmc->Gspos("B045", 14, "B077", dx, dy, -dz2, idrotm[2030], "ONLY");

  vmc->Gspos("B045", 15, "B077", -dx, -dy, dz2, idrotm[2024], "ONLY");
  vmc->Gspos("B045", 16, "B077", -dx, -dy, -dz2, idrotm[2032], "ONLY");
  vmc->Gspos("B045", 17, "B077", dx, -dy, dz2, idrotm[2025], "ONLY");
  vmc->Gspos("B045", 18, "B077", dx, -dy, -dz2, idrotm[2033], "ONLY");

  vmc->Gspos("BM49", 3, "B077", dx, -dy, 0., idrotm[2025], "ONLY");
  vmc->Gspos("BM49", 4, "B077", -dx, -dy, 0., idrotm[2024], "ONLY");

  //
  // The inner frame
  //
  //
  //  Mother Volumes
  //
  ptrd1[0] = (hR - longH / 2. - dext) * tan10;
  ptrd1[1] = (hR - longH / 2. + iFrH0) * tan10;
  ptrd1[2] = zE;
  ptrd1[3] = iFrH0 / 2.;
  float dd = longW / 2. * cos10 + 0.1;
  TGeoTrd1* shTRD1 = new TGeoTrd1("shTRD1", ptrd1[0], ptrd1[1], ptrd1[2], ptrd1[3]);
  TGeoBBox* shBox = new TGeoBBox("shBox", 50., zE + 10., 1.);
  TGeoRotation* rot1 = new TGeoRotation("urot1", 100., 0., 90., 90., 10., 0.);
  TGeoRotation* rot2 = new TGeoRotation("urot2", 80., 0., 90., 90., -10., 0.);
  float trotDz = iFrH0 / 2. + 1.;
  float trotDx = 402. * tan10;
  TGeoCombiTrans* trot1 = new TGeoCombiTrans(-trotDx, 0., trotDz, rot2);
  TGeoCombiTrans* trot2 = new TGeoCombiTrans(+trotDx, 0., trotDz, rot1);
  TGeoUnion* uni = new TGeoUnion(shBox, shBox, trot1, trot2);
  TGeoCompositeShape* shU = new TGeoCompositeShape("shU", uni);
  TGeoSubtraction* sub = new TGeoSubtraction(shTRD1, shU, nullptr, nullptr);
  TGeoCompositeShape* shCS = new TGeoCompositeShape("shCS", sub);
  // center of segments
  float r = (hR - longH / 2. + iFrH0 / 2.) - dext;
  // center of outer frame
  // vertical
  float rout1 = 406.0;
  // radial
  float rout2 = 412.3 - 2. * sin10 + 0.25;
  //
  TString module[18];
  for (i = 0; i < 18; i++) {
    // Create volume i
    char name[16];
    // official module numbering
    int mod = i + 13;
    if (mod > 17)
      mod -= 18;
    snprintf(name, 16, "BSEGMO%d", mod);
    //
    TGeoVolume* voTRD1 = new TGeoVolume(name, shCS, kMedAir);
    module[i] = name;
    // Place volume i
    float phi1 = i * 20.;
    float phi2 = 270. + phi1;
    if (phi2 >= 360.)
      phi2 -= 360.;
    dx = TMath::Sin(phi1 * kdeg2rad) * r;
    dy = -TMath::Cos(phi1 * kdeg2rad) * r;

    char nameR[16];
    snprintf(nameR, 16, "B43_Rot_%d", i);
    TGeoRotation* rot = new TGeoRotation(nameR, 90.0, phi1, 0., 0., 90., phi2);
    AliMatrix(idrotm[2034 + i], 90.0, phi1, 0., 0., 90., phi2);
    TGeoVolume* vol77 = gGeoManager->GetVolume("B077");
    vol77->AddNode(voTRD1, 1, new TGeoCombiTrans(dx, dy, 0., rot));

    //
    //    Position elements of outer Frame
    //
    dx = TMath::Sin(phi1 * kdeg2rad) * rout1;
    dy = -TMath::Cos(phi1 * kdeg2rad) * rout1;
    for (j = 0; j < 3; j++) {
      dz = dymodU[j];
      TGeoVolume* vol = gGeoManager->GetVolume("B077");
      vol->AddNode(asBI42, 6 * i + 2 * j + 1, new TGeoCombiTrans(dx, dy, dz, rot));
      vol->AddNode(asBI42, 6 * i + 2 * j + 2, new TGeoCombiTrans(dx, dy, -dz, rot));
    }

    phi1 = i * 20. + 10;
    phi2 = 270 + phi1;
    AliMatrix(idrotm[2052 + i], 90.0, phi1, 90., phi2, 0., 0.);

    dx = TMath::Sin(phi1 * kdeg2rad) * rout2;
    dy = -TMath::Cos(phi1 * kdeg2rad) * rout2;
    vmc->Gspos("B033", i + 1, "B077", dx, dy, 0., idrotm[2052 + i], "ONLY");
    //
  }
  // Internal Frame rings
  //
  //
  // Pos 7   60x60x5x6  for inner rings (I-beam)
  // Pos 6    100x60x5  for front and rear rings
  //
  // Front and rear
  //

  ptrd1[0] = (hR - longH / 2.) * tan10 - dd;
  ptrd1[1] = (hR + longH / 2.) * tan10 - dd;
  ptrd1[2] = ringW / 2.;
  ptrd1[3] = ringH / 2.;

  vmc->Gsvolu("B072", "TRD1", kSteel, ptrd1, 4);

  ptrd1[0] = (hR - longH / 2. + 0.5) * tan10 - dd;
  ptrd1[1] = (hR + longH / 2. - 0.5) * tan10 - dd;
  ptrd1[2] = ringW / 2. - 0.5;
  ptrd1[3] = ringH / 2. - 0.5;

  vmc->Gsvolu("B073", "TRD1", kAir, ptrd1, 4);
  vmc->Gspos("B073", 1, "B072", 0., 0., 0., 0, "ONLY");
  //
  // I-Beam
  // Mother volume
  TGeoVolumeAssembly* asBI72 = new TGeoVolumeAssembly("BI72");
  // Horizontal
  float rIB1 = hR + ringH / 2.;
  float rIB2 = hR - ringH / 2.;
  ptrd1[0] = (rIB1 - ringT / 2.) * tan10 - dd;
  ptrd1[1] = (rIB1)*tan10 - dd;
  ptrd1[2] = ringH / 2.;
  ptrd1[3] = ringT / 4.;
  vmc->Gsvolu("BIH172", "TRD1", kSteel, ptrd1, 4);
  ptrd1[0] = (rIB2)*tan10 - dd;
  ptrd1[1] = (rIB2 + ringT / 2.) * tan10 - dd;
  ptrd1[2] = ringH / 2.;
  ptrd1[3] = ringT / 4.;
  vmc->Gsvolu("BIH272", "TRD1", kSteel, ptrd1, 4);

  // Vertical
  ptrd1[0] = (rIB2 + ringT / 2.) * tan10 - dd;
  ptrd1[1] = (rIB1 - ringT / 2.) * tan10 - dd;
  ptrd1[2] = 0.6 / 2.;
  ptrd1[3] = (ringH - ringT) / 2.;
  vmc->Gsvolu("BIV72", "TRD1", kSteel, ptrd1, 4);
  // Place
  asBI72->AddNode(gGeoManager->GetVolume("BIV72"), 1, new TGeoTranslation(0., 0., 0.));
  asBI72->AddNode(gGeoManager->GetVolume("BIH172"), 1, new TGeoTranslation(0., 0., (ringH / 2. - ringT / 4.)));
  asBI72->AddNode(gGeoManager->GetVolume("BIH272"), 1, new TGeoTranslation(0., 0., -(ringH / 2. - ringT / 4.)));

  // Web frame
  //
  // h x w x s = 60 x 40 x 5
  // (attention: elements are half bars, "U" shaped)
  //

  createWebFrame("B063", dHz, dymodO[0], 10.);
  createWebFrame("B163", dHz, dymodO[1], 10.);
  createWebFrame("B263", dHz, dymodO[2], 10.);
  createWebFrame("B363", dHz, dymodO[3], 10.);
  createWebFrame("B463", dHz, dymodO[4], 10.);

  dz = -iFrH0 / 2. + ringH / 2. + dext;

  float dz0 = -iFrH0 / 2. + longH + 113. / 2. + dext - 0.1;
  float dx0 = (hR + iFrH / 2.) * tan10 - longW / 4. * cos10 - 0.065;
  for (jmod = 0; jmod < 18; jmod++) {
    //
    // ring bars
    for (i = 0; i < 3; i++) {
      if (i == 2) {
        vmc->Gspos("B072", 6 * jmod + i + 1, module[jmod], 0, dymodL[i], dz, 0, "ONLY");
        vmc->Gspos("B072", 6 * jmod + i + 4, module[jmod], 0, -dymodL[i], dz, idrotm[2070], "ONLY");
      } else {
        TGeoVolume* vol = gGeoManager->GetVolume(module[jmod]);
        vol->AddNode(asBI72, 6 * jmod + i + 1, new TGeoTranslation(0, dymodL[i], dz));
        vol->AddNode(asBI72, 6 * jmod + i + 4, new TGeoTranslation(0, -dymodL[i], dz));
      }
    }
  }
  //
  // outer diagonal web

  dy = dymodW[0] - (dHz / 2.) * TMath::Tan(dymodO[0] * kdeg2rad);

  for (jmod = 0; jmod < 18; jmod++) {
    vmc->Gspos("B063I", 4 * jmod + 1, module[jmod], dx0, dy, dz0, idrotm[2096], "ONLY");
    vmc->Gspos("B063", 4 * jmod + 2, module[jmod], dx0, -dy, dz0, idrotm[2097], "ONLY");
    vmc->Gspos("B063I", 4 * jmod + 3, module[jmod], -dx0, -dy, dz0, idrotm[2087], "ONLY");
    vmc->Gspos("B063", 4 * jmod + 4, module[jmod], -dx0, dy, dz0, idrotm[2086], "ONLY");
  }

  dy = dymodW[1] - (dHz / 2.) * TMath::Tan(dymodO[1] * kdeg2rad);

  for (jmod = 0; jmod < 18; jmod++) {
    vmc->Gspos("B163I", 4 * jmod + 1, module[jmod], dx0, -dy, dz0, idrotm[2096], "ONLY");
    vmc->Gspos("B163", 4 * jmod + 2, module[jmod], dx0, dy, dz0, idrotm[2097], "ONLY");
    vmc->Gspos("B163I", 4 * jmod + 3, module[jmod], -dx0, dy, dz0, idrotm[2087], "ONLY");
    vmc->Gspos("B163", 4 * jmod + 4, module[jmod], -dx0, -dy, dz0, idrotm[2086], "ONLY");
  }

  dy = dymodW[2] - (dHz / 2) * TMath::Tan(dymodO[2] * kdeg2rad);

  for (jmod = 0; jmod < 18; jmod++) {
    vmc->Gspos("B263I", 4 * jmod + 1, module[jmod], dx0, dy, dz0, idrotm[2096], "ONLY");
    vmc->Gspos("B263", 4 * jmod + 2, module[jmod], dx0, -dy, dz0, idrotm[2097], "ONLY");
    vmc->Gspos("B263I", 4 * jmod + 3, module[jmod], -dx0, -dy, dz0, idrotm[2087], "ONLY");
    vmc->Gspos("B263", 4 * jmod + 4, module[jmod], -dx0, dy, dz0, idrotm[2086], "ONLY");
  }

  dy = dymodW[3] - (dHz / 2.) * TMath::Tan(dymodO[3] * kdeg2rad);

  for (jmod = 0; jmod < 18; jmod++) {
    vmc->Gspos("B363I", 4 * jmod + 1, module[jmod], dx0, -dy, dz0, idrotm[2096], "ONLY");
    vmc->Gspos("B363", 4 * jmod + 2, module[jmod], dx0, dy, dz0, idrotm[2097], "ONLY");
    vmc->Gspos("B363I", 4 * jmod + 3, module[jmod], -dx0, dy, dz0, idrotm[2087], "ONLY");
    vmc->Gspos("B363", 4 * jmod + 4, module[jmod], -dx0, -dy, dz0, idrotm[2086], "ONLY");
  }

  dy = dymodW[4] - (dHz / 2.) * TMath::Tan(dymodO[4] * kdeg2rad);

  for (jmod = 0; jmod < 18; jmod++) {
    vmc->Gspos("B463I", 4 * jmod + 1, module[jmod], dx0, dy, dz0, idrotm[2096], "ONLY");
    vmc->Gspos("B463", 4 * jmod + 2, module[jmod], dx0, -dy, dz0, idrotm[2097], "ONLY");
    vmc->Gspos("B463I", 4 * jmod + 3, module[jmod], -dx0, -dy, dz0, idrotm[2087], "ONLY");
    vmc->Gspos("B463", 4 * jmod + 4, module[jmod], -dx0, dy, dz0, idrotm[2086], "ONLY");
  }

  // longitudinal bars (TPC rails attached)
  //  new specs:
  //  h x w x s = 100 x 75 x 6
  //  Attention: 2 "U" shaped half rods per cell
  //  longitudinal bars (no TPC rails attached)
  //  new specs: h x w x s = 40 x 60 x 5
  //
  //
  //
  double lbox[3];
  lbox[0] = longW / 4.;
  lbox[2] = longH / 2.;
  lbox[1] = longLI / 2.;
  vmc->Gsvolu("BA59", "BOX", kSteel, lbox, 3);
  gGeoManager->GetVolume("BA59")->SetVisContainers();
  lbox[0] = longW / 4. - 0.25;
  lbox[2] = longH / 2. - 0.50;
  vmc->Gsvolu("BA62", "BOX", kAir, lbox, 3);
  vmc->Gspos("BA62", 1, "BA59", 0.25, 0.0, 0.0, 0, "ONLY");

  dz = -iFrH0 / 2. + longH / 2. - 1. * sin10 + dext;
  dx = hR * tan10 - longW / 4. * cos10 - 0.065;
  for (jmod = 0; jmod < 18; jmod++) {
    vmc->Gspos("BA59", 2 * jmod + 1, module[jmod], dx, 0.0, dz, idrotm[2096], "ONLY");
    vmc->Gspos("BA59", 2 * jmod + 2, module[jmod], -dx, 0.0, dz, idrotm[2087], "ONLY");
  }

  //
  // Rails for TRD
  //
  // Pos 1
  //
  // angular 80 deg profile
  lbox[2] = 4.0;
  lbox[0] = 0.2;
  lbox[1] = longLI / 2.;
  vmc->Gsvolu("BTRDR_10", "BOX", kSteel, lbox, 3);

  ptrd1[0] = 3.;
  ptrd1[1] = 3. + 0.4 * tan10;
  ptrd1[2] = longLI / 2.;
  ptrd1[3] = 0.2;
  vmc->Gsvolu("BTRDR_11", "TRD1", kSteel, ptrd1, 4);

  lbox[2] = 2.0;
  lbox[0] = 0.3;
  lbox[1] = longLI / 2.;
  vmc->Gsvolu("BTRDR_12", "BOX", kAlu, lbox, 3);
  gGeoManager->GetVolume("BTRDR_12")->SetVisContainers();

  lbox[2] = 2.0;
  lbox[0] = 0.1;
  lbox[1] = longLI / 2.;
  vmc->Gsvolu("BTRDR_13", "BOX", kG10, lbox, 3);
  vmc->Gspos("BTRDR_13", 1, "BTRDR_12", -0.2, 0.0, 0.0, 0, "ONLY");

  lbox[2] = 0.1;
  lbox[0] = 2.0;
  lbox[1] = longLI / 2.;
  vmc->Gsvolu("BTRDR_14", "BOX", kG10, lbox, 3);
  dz = -iFrH0 / 2. + longH / 2. + dext;
  float zpos = 80.;
  int isec_1[11] = {0, 1, 2, 3, 4, 5, 13, 14, 15, 16, 17};

  for (int index = 0; index < 11; index++) {
    jmod = isec_1[index];
    float dz1 = dz + 3. + (zpos - 4.);
    dx0 = (hR + dz0 + zpos - 4.) * tan10 - (longW / 2. + 0.2) / cos10 - 0.05;
    if (jmod != 5)
      vmc->Gspos("BTRDR_10", 2 * jmod + 1, module[jmod], dx0, 0.0, dz1, idrotm[2096], "ONLY");
    if (jmod != 13)
      vmc->Gspos("BTRDR_10", 2 * jmod + 2, module[jmod], -dx0, 0.0, dz1, idrotm[2086], "ONLY");
    dx0 -= 0.5;
    if (jmod != 5)
      vmc->Gspos("BTRDR_12", 2 * jmod + 1, module[jmod], dx0, 0.0, dz1, idrotm[2096], "ONLY");
    if (jmod != 13)
      vmc->Gspos("BTRDR_12", 2 * jmod + 2, module[jmod], -dx0, 0.0, dz1, idrotm[2087], "ONLY");
    dz1 += (4 - 0.2);
    dz1 += dext;
    dx0 = (hR + dz0 + zpos - 0.2) * tan10 - (longW / 2. + 3. + 0.4) / cos10;
    if (jmod != 5)
      vmc->Gspos("BTRDR_11", 2 * jmod + 1, module[jmod], dx0, 0.0, dz1, 0, "ONLY");
    if (jmod != 13)
      vmc->Gspos("BTRDR_11", 2 * jmod + 2, module[jmod], -dx0, 0.0, dz1, 0, "ONLY");
    dz1 -= 0.3;
    dx0 -= 0.5;
    if (jmod != 5)
      vmc->Gspos("BTRDR_14", 2 * jmod + 1, module[jmod], dx0, 0.0, dz1, 0, "ONLY");
    if (jmod != 13)
      vmc->Gspos("BTRDR_14", 2 * jmod + 2, module[jmod], -dx0, 0.0, dz1, 0, "ONLY");
  }

  // Pos 2
  // 40 x 10
  lbox[2] = 2.0;
  lbox[0] = 0.5;
  lbox[1] = longLI / 2.;
  vmc->Gsvolu("BTRDR_2", "BOX", kAlu, lbox, 3);
  lbox[2] = 2.0;
  lbox[0] = 0.1;
  lbox[1] = longLI / 2.;
  vmc->Gsvolu("BTRDR_21", "BOX", kG10, lbox, 3);
  vmc->Gspos("BTRDR_21", 1, "BTRDR_2", -0.4, 0.0, 0.0, 0, "ONLY");

  int isec_2a[16] = {1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17};
  for (int index = 0; index < 16; index++) {
    jmod = isec_2a[index];
    dx0 = (hR + dz0) * tan10 + 10. * sin10 - (longW / 4. + 0.5) / cos10;
    if (jmod > 8) {
      vmc->Gspos("BTRDR_2", 2 * jmod + 1, module[jmod], dx0 - 1.5, 0.0, dz + 3. + 8. * cos10, idrotm[2096], "ONLY");
    } else {
      vmc->Gspos("BTRDR_2", 2 * jmod + 2, module[jmod], -dx0 + 1.5, 0.0, dz + 3. + 8. * cos10, idrotm[2087], "ONLY");
    }
  }

  int isec_2b[6] = {6, 7, 8, 10, 11, 12};
  for (int index = 0; index < 6; index++) {
    jmod = isec_2b[index];
    dx0 = (hR + dz0 + zpos - 3.) * tan10 - (longW / 4. + 0.5) / cos10;
    if (index < 3) {
      vmc->Gspos("BTRDR_2", 2 * jmod + 2, module[jmod], -dx0 + 1.5, 0.0, dz + 3. + zpos - 3., idrotm[2087], "ONLY");
    } else {
      vmc->Gspos("BTRDR_2", 2 * jmod + 1, module[jmod], dx0 - 1.5, 0.0, dz + 3. + zpos - 3., idrotm[2096], "ONLY");
    }
  }

  // Pos 3
  // 40 x 14
  lbox[0] = 2.0;
  lbox[2] = 0.7;
  lbox[1] = longLI / 2.;
  vmc->Gsvolu("BTRDR_3", "BOX", kAlu, lbox, 3);

  lbox[0] = 2.0;
  lbox[2] = 0.1;
  lbox[1] = longLI / 2.;
  vmc->Gsvolu("BTRDR_31", "BOX", kG10, lbox, 3);
  vmc->Gspos("BTRDR_31", 1, "BTRDR_3", 0, 0.0, 0.6, 0, "ONLY");

  int isec_3[9] = {5, 6, 7, 8, 9, 10, 11, 12, 13};

  for (int index = 0; index < 9; index++) {
    jmod = isec_3[index];
    if (index > 1)
      vmc->Gspos("BTRDR_3", 2 * jmod + 1, module[jmod], 50.96 - 5 - 2., 0.0, dz + 3.7, 0, "ONLY");
    if (index < 7)
      vmc->Gspos("BTRDR_3", 2 * jmod + 2, module[jmod], -50.96 + 5 + 2., 0.0, dz + 3.7, 0, "ONLY");
  }
  //
  // Fixation Blocks with tie anchors
  //
  // inner
  float thetFB1 = 10. / 180. * TMath::Pi();
  float thetFB2 = 40. / 180. * TMath::Pi();
  // half height of the block
  double dzFB = 6.;
  // half width of the block
  float dyFB = 3.9 / 2.;
  // lenth upper face
  float dxFB = 46.;
  // lower face
  float dx1FB = dxFB / 2. - 2. * dzFB * TMath::Tan(thetFB1);
  float dx2FB = dxFB / 2. - 2. * dzFB * TMath::Tan(thetFB2);

  TGeoArb8* shFB1 = new TGeoArb8(dzFB);
  shFB1->SetVertex(0, -dyFB / 2., -dxFB / 2.);
  shFB1->SetVertex(1, -dyFB / 2., dxFB / 2.);
  shFB1->SetVertex(2, dyFB / 2., dxFB / 2.);
  shFB1->SetVertex(3, dyFB / 2., -dxFB / 2.);

  shFB1->SetVertex(4, -dyFB / 2., -dx1FB);
  shFB1->SetVertex(5, -dyFB / 2., dx2FB);
  shFB1->SetVertex(6, dyFB / 2., dx2FB);
  shFB1->SetVertex(7, dyFB / 2., -dx1FB);

  TGeoVolume* volFB1 = new TGeoVolume("BTRD_FB1", shFB1, kMedAlu);
  //
  // tie anchors rectangular profile 4 x 8
  TGeoVolume* volTAR11 = new TGeoVolume("BTRD_TAR11", new TGeoBBox(dyFB / 2., dxFB / 2. - 0.7, 4.), kMedAlu);
  TGeoVolume* volTAR12 = new TGeoVolume("BTRD_TAR12", new TGeoBBox(dyFB / 2. - 0.25, dxFB / 2., 3. - 0.5), kMedAir);
  volTAR11->AddNode(volTAR12, 1, new TGeoTranslation(0.25, 0., 0.0));
  // clamp (about twice the length of the block), 6 mm thick (read off from a foto)
  TGeoVolume* volTAR13 = new TGeoVolume("BTRD_TAR13", new TGeoBBox(0.3, 45., 3.), kMedAlu);
  // square block with screw r = 0.9 cm
  TGeoVolume* volTAR141 = new TGeoVolume("BTRD_TAR141", new TGeoBBox(1., 2., 6.), kMedAir);
  TGeoVolume* volTAR142 = new TGeoVolume("BTRD_TAR142", new TGeoBBox(1., 2., 6.), kMedAir);

  TGeoVolume* volTAR15 = new TGeoVolume("BTRD_TAR15", new TGeoBBox(1., 2., 3.), kMedAlu);
  TGeoVolume* volTAR16 = new TGeoVolume("BTRD_TAR16", new TGeoTubeSeg(0., 0.78, 1.5, 90., 270.), kMedSteel);
  TGeoVolume* volTAR17 = new TGeoVolume("BTRD_TAR17", new TGeoTubeSeg(0., 0.78, 1.5, -90, 90.), kMedSteel);
  volTAR141->AddNode(volTAR15, 1, new TGeoTranslation(0, 0, 0));
  volTAR141->AddNode(volTAR16, 1, new TGeoTranslation(1., 0, +4.5));
  volTAR141->AddNode(volTAR16, 2, new TGeoTranslation(1., 0, -4.5));

  volTAR142->AddNode(volTAR15, 1, new TGeoTranslation(0, 0, 0));
  volTAR142->AddNode(volTAR17, 1, new TGeoTranslation(-1., 0, +4.5));
  volTAR142->AddNode(volTAR17, 2, new TGeoTranslation(-1., 0, -4.5));

  TGeoVolumeAssembly* asFB1 = new TGeoVolumeAssembly("BTRD_FBAS1");
  TGeoVolumeAssembly* asFB2 = new TGeoVolumeAssembly("BTRD_FBAS2");
  asFB1->AddNode(volFB1, 1, gGeoIdentity);
  asFB1->AddNode(volTAR11, 1, new TGeoTranslation(0., 0., -dzFB - 3.));
  asFB1->AddNode(volTAR13, 1, new TGeoTranslation(-1.36, 4.5, -dzFB - 3.));
  asFB1->AddNode(volTAR141, 1, new TGeoTranslation(0., dxFB - 2. + 4.5, -dzFB - 3.));
  asFB1->AddNode(volTAR141, 2, new TGeoTranslation(0., -dxFB + 2. + 4.5, -dzFB - 3.));

  asFB2->AddNode(volFB1, 2, gGeoIdentity);
  asFB2->AddNode(volTAR11, 2, new TGeoTranslation(0., 0., -dzFB - 3.));
  asFB2->AddNode(volTAR13, 2, new TGeoTranslation(1.36, 4.5, -dzFB - 3.));
  asFB2->AddNode(volTAR142, 3, new TGeoTranslation(0., dxFB - 2. + 4.5, -dzFB - 3.));
  asFB2->AddNode(volTAR142, 4, new TGeoTranslation(0., -dxFB + 2. + 4.5, -dzFB - 3.));
  //
  // Fixation block outer
  //
  thetFB1 = 20. / 180. * TMath::Pi();
  thetFB2 = 27. / 180. * TMath::Pi();

  dxFB = 42.0;

  dx1FB = dxFB / 2. - 2. * dzFB * TMath::Tan(thetFB1);
  dx2FB = dxFB / 2. - 2. * dzFB * TMath::Tan(thetFB2);

  TGeoArb8* shFB2 = new TGeoArb8(dzFB);
  shFB2->SetVertex(0, -dyFB / 2., -dxFB / 2.);
  shFB2->SetVertex(1, -dyFB / 2., dxFB / 2.);
  shFB2->SetVertex(2, dyFB / 2., dxFB / 2.);
  shFB2->SetVertex(3, dyFB / 2., -dxFB / 2.);

  shFB2->SetVertex(4, -dyFB / 2., -dx1FB);
  shFB2->SetVertex(5, -dyFB / 2., dx2FB);
  shFB2->SetVertex(6, dyFB / 2., dx2FB);
  shFB2->SetVertex(7, dyFB / 2., -dx1FB);

  TGeoVolume* volFB2 = new TGeoVolume("BTRD_FB2", shFB2, kMedAlu);
  TGeoVolume* volTAR21 = new TGeoVolume("BTRD_TAR21", new TGeoBBox(dyFB / 2., dxFB / 2., 3.), kMedAlu);
  TGeoVolume* volTAR22 = new TGeoVolume("BTRD_TAR22", new TGeoBBox(dyFB / 2. - 0.25, dxFB / 2., 3. - 0.5), kMedAir);
  volTAR21->AddNode(volTAR22, 1, new TGeoTranslation(-0.25, 0., 0.0));
  // tie anchor
  TGeoVolume* volTAR23 = new TGeoVolume("BTRD_TAR23", new TGeoBBox(0.3, 40., 3.), kMedAlu);

  TGeoVolumeAssembly* asFB3 = new TGeoVolumeAssembly("BTRD_FBAS3");
  TGeoVolumeAssembly* asFB4 = new TGeoVolumeAssembly("BTRD_FBAS4");
  asFB3->AddNode(volFB2, 1, gGeoIdentity);
  asFB3->AddNode(volTAR21, 1, new TGeoTranslation(0., 0., -dzFB - 3.));
  asFB3->AddNode(volTAR23, 1, new TGeoTranslation(-1.36, 0., -dzFB - 3.));
  asFB3->AddNode(volTAR141, 1, new TGeoTranslation(0., dxFB - 2., -dzFB - 3.));
  asFB3->AddNode(volTAR141, 2, new TGeoTranslation(0., -dxFB + 2., -dzFB - 3.));

  asFB4->AddNode(volFB2, 2, gGeoIdentity);
  asFB4->AddNode(volTAR21, 2, new TGeoTranslation(0., 0., -dzFB - 3.));
  asFB4->AddNode(volTAR23, 2, new TGeoTranslation(1.36, 0.5, -dzFB - 3.));
  asFB4->AddNode(volTAR142, 3, new TGeoTranslation(0., dxFB - 2. + 0.5, -dzFB - 3.));
  asFB4->AddNode(volTAR142, 4, new TGeoTranslation(0., -dxFB + 2. + 0.5, -dzFB - 3.));

  float zTA1 = 21.1;
  float yFB1 = 87.6;
  float yFB2 = 231.4;
  dx = ((hR - longH / 2. + iFrH0 / 2.) - dext + zTA1) * tan10 - 3.9 / 4.;

  for (int index = 0; index < 11; index++) {
    int imod = isec_1[index];
    if (imod != 5)
      vmc->Gspos("BTRD_FBAS2", index, module[imod], dx, -yFB1, zTA1, idrotm[2097], "ONLY");

    if (imod != 13)
      vmc->Gspos("BTRD_FBAS1", 11 + index, module[imod], -dx, -yFB1, zTA1, idrotm[2087], "ONLY");

    if (imod != 5)
      vmc->Gspos("BTRD_FBAS1", 22 + index, module[imod], dx, yFB1, zTA1, idrotm[2096], "ONLY");

    if (imod != 13)
      vmc->Gspos("BTRD_FBAS2", 33 + index, module[imod], -dx, yFB1, zTA1, idrotm[2086], "ONLY");

    if (imod != 5)
      vmc->Gspos("BTRD_FBAS4", index, module[imod], dx, -yFB2, zTA1, idrotm[2097], "ONLY");

    if (imod != 13)
      vmc->Gspos("BTRD_FBAS3", 11 + index, module[imod], -dx, -yFB2, zTA1, idrotm[2087], "ONLY");

    if (imod != 5)
      vmc->Gspos("BTRD_FBAS3", 22 + index, module[imod], dx, yFB2, zTA1, idrotm[2096], "ONLY");

    if (imod != 13)
      vmc->Gspos("BTRD_FBAS4", 33 + index, module[imod], -dx, yFB2, zTA1, idrotm[2086], "ONLY");
  }

  //
  // TOF Support Structures

  //
  // Frame extension rectangular beams
  lbox[0] = 6;
  lbox[1] = 3.;
  lbox[2] = 36.0;
  TGeoVolume* voBTOFS1 = new TGeoVolume("BTOFS1", new TGeoBBox(lbox), gGeoManager->GetMedium("FRAME_Steel"));
  lbox[0] = 5.5;
  lbox[1] = 2.5;
  lbox[2] = 36.0;
  TGeoVolume* voBTOFS11 = new TGeoVolume("BTOFS11", new TGeoBBox(lbox), gGeoManager->GetMedium("FRAME_Air"));
  voBTOFS1->AddNode(voBTOFS11, 1, gGeoIdentity);

  //
  // Frame extension rectangular beams
  // upper clamps
  TGeoXtru* shBTOFS2 = new TGeoXtru(2);
  TGeoXtru* shBTOFS3 = new TGeoXtru(2);
  TGeoXtru* shBTOFS4 = new TGeoXtru(2);
  TGeoXtru* shBTOFS5 = new TGeoXtru(2);

  double xxtru1[7];
  double yxtru1[7];
  // 1
  xxtru1[0] = 8.5;
  yxtru1[0] = 4.5;
  // 2
  xxtru1[1] = -6.0;
  yxtru1[1] = 4.5;
  // 3
  xxtru1[2] = -8.5;
  yxtru1[2] = 4.5 - 2.5 * sin10;
  // 4
  xxtru1[3] = 8.5 - 14.5 / cos10;
  yxtru1[3] = -6. - 14.5 * sin10;
  // 5
  xxtru1[4] = 8.5 - 10.5 / cos10;
  yxtru1[4] = -6. - 10.5 * sin10;
  // 6
  xxtru1[5] = xxtru1[4] + 8. * sin10;
  yxtru1[5] = yxtru1[4] - 8. / cos10;
  // 7
  xxtru1[6] = 8.5;
  yxtru1[6] = -6.0;

  double xxtru2[7];
  for (i = 0; i < 7; i++)
    xxtru2[i] = -xxtru1[i];

  double xxtru3[5];
  double yxtru3[5];
  double xxtru4[5];
  for (i = 0; i < 4; i++) {
    xxtru3[i] = xxtru1[i];
    yxtru3[i] = yxtru1[i];
  }
  xxtru3[4] = xxtru1[6];
  yxtru3[4] = yxtru1[6];
  for (i = 0; i < 5; i++)
    xxtru4[i] = -xxtru3[i];

  shBTOFS2->DefinePolygon(7, xxtru1, yxtru1);
  shBTOFS2->DefineSection(0, -4.);
  shBTOFS2->DefineSection(1, +4.);

  shBTOFS3->DefinePolygon(7, xxtru2, yxtru1);
  shBTOFS3->DefineSection(0, -4.);
  shBTOFS3->DefineSection(1, +4.);
  TGeoVolume* voBTOFS2 = new TGeoVolume("BTOFS2", shBTOFS2, gGeoManager->GetMedium("FRAME_Steel"));
  TGeoVolume* voBTOFS3 = new TGeoVolume("BTOFS3", shBTOFS3, gGeoManager->GetMedium("FRAME_Steel"));

  // different fixation for clamps close to web frame
  shBTOFS4->DefinePolygon(5, xxtru3, yxtru3);
  shBTOFS4->DefineSection(0, -4.);
  shBTOFS4->DefineSection(1, +4.);

  shBTOFS5->DefinePolygon(5, xxtru4, yxtru3);
  shBTOFS5->DefineSection(0, -4.);
  shBTOFS5->DefineSection(1, +4.);
  TGeoVolume* voBTOFS4 = new TGeoVolume("BTOFS4", shBTOFS4, gGeoManager->GetMedium("FRAME_Steel"));
  TGeoVolume* voBTOFS5 = new TGeoVolume("BTOFS5", shBTOFS5, gGeoManager->GetMedium("FRAME_Steel"));

  lbox[0] = 5.5;
  lbox[1] = 2.5;
  lbox[2] = 4.0;
  TGeoVolume* voBTOFS21 = new TGeoVolume("BTOFS21", new TGeoBBox(lbox), gGeoManager->GetMedium("FRAME_Air"));
  voBTOFS2->AddNode(voBTOFS21, 1, gGeoIdentity);
  voBTOFS3->AddNode(voBTOFS21, 2, gGeoIdentity);
  voBTOFS4->AddNode(voBTOFS21, 3, gGeoIdentity);
  voBTOFS5->AddNode(voBTOFS21, 4, gGeoIdentity);

  TGeoVolumeAssembly* asTOFS00 = new TGeoVolumeAssembly("BTOFS00");
  asTOFS00->AddNode(voBTOFS1, 1, gGeoIdentity);
  asTOFS00->AddNode(voBTOFS2, 1, new TGeoTranslation(0., 0., 40.));
  asTOFS00->AddNode(voBTOFS2, 2, new TGeoTranslation(0., 0., -40.));

  TGeoVolumeAssembly* asTOFS01 = new TGeoVolumeAssembly("BTOFS01");
  asTOFS01->AddNode(voBTOFS1, 2, gGeoIdentity);
  asTOFS01->AddNode(voBTOFS3, 1, new TGeoTranslation(0., 0., 40.));
  asTOFS01->AddNode(voBTOFS3, 2, new TGeoTranslation(0., 0., -40.));

  TGeoVolumeAssembly* asTOFS02 = new TGeoVolumeAssembly("BTOFS02");
  asTOFS02->AddNode(voBTOFS1, 3, gGeoIdentity);
  asTOFS02->AddNode(voBTOFS2, 3, new TGeoTranslation(0., 0., -40.));
  asTOFS02->AddNode(voBTOFS4, 2, new TGeoTranslation(0., 0., 40.));

  TGeoVolumeAssembly* asTOFS03 = new TGeoVolumeAssembly("BTOFS03");
  asTOFS03->AddNode(voBTOFS1, 4, gGeoIdentity);
  asTOFS03->AddNode(voBTOFS3, 3, new TGeoTranslation(0., 0., -40.));
  asTOFS03->AddNode(voBTOFS5, 2, new TGeoTranslation(0., 0., 40.));

  asTOFS00->SetVisibility(1);
  asTOFS01->SetVisibility(1);

  for (i = 0; i < 18; i++) {
    float phi1 = i * 20.;
    float phi2 = 270. + phi1;
    rot1 = new TGeoRotation(Form("TOFS_R1_%d", i), 90.0, phi1, 90., phi2, 0., 0.);
    dx = TMath::Sin((phi1 + 8.95) * kdeg2rad) * (rout2 + 12.);
    dy = -TMath::Cos((phi1 + 8.95) * kdeg2rad) * (rout2 + 12.);
    if ((i > 3 && i < 8) || (i > 10 && i < 15)) {
      (gGeoManager->GetVolume("B077"))->AddNode(asTOFS03, i, new TGeoCombiTrans(dx, dy, 345. - 53. - 0.5, rot1));
    } else {
      (gGeoManager->GetVolume("B077"))->AddNode(asTOFS01, i, new TGeoCombiTrans(dx, dy, 345. - 53. - 0.5, rot1));
    }
    dx = TMath::Sin((phi1 - 8.95) * kdeg2rad) * (rout2 + 12.);
    dy = -TMath::Cos((phi1 - 8.95) * kdeg2rad) * (rout2 + 12.);
    if ((i > 3 && i < 8) || (i > 10 && i <= 15)) {
      (gGeoManager->GetVolume("B077"))->AddNode(asTOFS02, i, new TGeoCombiTrans(dx, dy, 345. - 53 - 0.5, rot1));
    } else {
      (gGeoManager->GetVolume("B077"))->AddNode(asTOFS00, i, new TGeoCombiTrans(dx, dy, 345. - 53 - 0.5, rot1));
    }
  }

  //
  // Thermal shield
  //

  float dyM = 99.0;
  makeHeatScreen("M", dyM, idrotm[2090], idrotm[2091]);
  float dyAM = 119.5;
  makeHeatScreen("AM", dyAM, idrotm[2090], idrotm[2091]);
  float dyA = 122.5 - 5.5;
  makeHeatScreen("A", dyA, idrotm[2090], idrotm[2091]);

  //
  //
  //
  dz = -57.2 + 0.6;
  for (i = 0; i < 18; i++) {
    char nameMo[16];
    snprintf(nameMo, 16, "BSEGMO%d", i);
    // M
    vmc->Gspos("BTSH_M", i + 1, nameMo, 0., 0., dz, 0, "ONLY");
    // AM, CM
    dy = dymodL[0] + dyAM / 2. + 3.;
    vmc->Gspos("BTSH_AM", i + 1, nameMo, 0., dy, dz, 0, "ONLY");
    vmc->Gspos("BTSH_AM", i + 19, nameMo, 0., -dy, dz, 0, "ONLY");
    // A, C
    dy = dymodL[1] + dyA / 2 + 0.4;
    vmc->Gspos("BTSH_A", i + 1, nameMo, 0., dy, dz, 0, "ONLY");
    vmc->Gspos("BTSH_A", i + 19, nameMo, 0., -dy, dz, 0, "ONLY");
  }

  //
  // TRD mother volumes
  //
  // absolute position of center 290.43 + 38.95 = 329.38
  // frame center                283.00 + 59.50 = 342.50
  // relative position of TRD    329.38 - 342.50
  //
  // shift wrt v2
  //
  const float zsh = -0.326;
  //
  ptrd1[0] = 47.4405; // CBL 28/6/2006
  ptrd1[1] = 61.1765; // CBL
  ptrd1[2] = 375.5;   // CBL
  ptrd1[3] = 38.95;   // CBL

  for (i = 0; i < 18; i++) {
    char nameCh[16];
    snprintf(nameCh, 16, "BTRD%d", i);
    char nameMo[16];
    snprintf(nameMo, 16, "BSEGMO%d", i);
    vmc->Gsvolu(nameCh, "TRD1", kAir, ptrd1, 4);
    gGeoManager->GetVolume(nameCh)->SetVisibility(false);
    vmc->Gspos(nameCh, 1, nameMo, 0., 0., -12.62 + zsh, 0, "ONLY"); // CBL 28/6/2006
  }

  //
  // TOF mother volumes as modified by B.Guerzoni
  // to remove overlaps/extrusions in case of aligned TOF SMs
  //
  ptrd1[0] = 62.2500;
  ptrd1[1] = 64.25;
  ptrd1[2] = 372.6;
  ptrd1[3] = 14.525 / 2;

  char nameChA[16];
  snprintf(nameChA, 16, "BTOFA");
  TGeoTrd1* trd1 = new TGeoTrd1(nameChA, ptrd1[0], ptrd1[1], ptrd1[2], ptrd1[3]);
  trd1->SetName("BTOFA"); // just to avoid a warning

  char nameChB[16];
  snprintf(nameChB, 16, "BTOFB");
  TGeoBBox* box1 = new TGeoBBox(nameChB, 64.25, 372.6, 1.0);
  box1->SetName("BTOFB"); // just to avoid a warning

  char nameChC[16];
  snprintf(nameChC, 16, "BTOFC");
  TGeoBBox* box2 = new TGeoBBox(nameChC, 62.25, 372.6, 6.2625);
  box2->SetName("BTOFC"); // just to avoid a warning

  TGeoTranslation* tr1 = new TGeoTranslation("trnsl1", 0, 0, -14.525 / 2);
  tr1->RegisterYourself();
  TGeoTranslation* tr2 = new TGeoTranslation("trnsl2", 0, 0, +1.0);
  tr2->RegisterYourself();
  TGeoTranslation* tr3 = new TGeoTranslation("trnsl3", 0, 0, 8.2625);
  tr3->RegisterYourself();
  TGeoCompositeShape* btofcs = new TGeoCompositeShape("Btofcs", "(BTOFA:trnsl1)+(BTOFB:trnsl2)+(BTOFC:trnsl3)");
  for (i = 0; i < 18; i++) {
    char nameCh[16];
    snprintf(nameCh, 16, "BTOF%d", i);
    char nameMo[16];
    snprintf(nameMo, 16, "BSEGMO%d", i);
    TGeoVolume* btf = new TGeoVolume(nameCh, btofcs, gGeoManager->GetMedium("FRAME_Air"));
    btf->SetName(nameCh);
    gGeoManager->GetVolume(nameCh)->SetVisibility(false);
    vmc->Gspos(nameCh, 1, nameMo, 0., 0., 43.525 + zsh, 0, "ONLY");
  }

  // Create TOF Rail
  char nameRaV1[16];
  snprintf(nameRaV1, 16, "RaV1");
  TGeoBBox* boxRaV1 = new TGeoBBox(nameRaV1, 0.5, 350.0, 1.5);
  char nameRaO1[16];
  snprintf(nameRaO1, 16, "RaO1");
  TGeoBBox* boxRaO1 = new TGeoBBox(nameRaO1, 1.5, 350.0, 0.5);
  TGeoCompositeShape* C1 = (TGeoCompositeShape*)createTOFRail(45.61);
  C1->SetName("C1");
  TGeoCompositeShape* C2 = (TGeoCompositeShape*)createTOFRail(61.61);
  C2->SetName("C2");
  TGeoCompositeShape* C3 = (TGeoCompositeShape*)createTOFRail(63.11);
  C3->SetName("C3");
  TGeoCompositeShape* C4 = (TGeoCompositeShape*)createTOFRail(61.61);
  C4->SetName("C4");
  TGeoCompositeShape* C5 = (TGeoCompositeShape*)createTOFRail(45.61);
  C5->SetName("C5");
  TGeoTranslation* trRaO1 = new TGeoTranslation("trRaO1", 1., 0., -2.);
  trRaO1->RegisterYourself();
  TGeoTranslation* trC1 = new TGeoTranslation("trC1", -3.39, -286.6, -0.15);
  trC1->RegisterYourself();
  TGeoTranslation* trC2 = new TGeoTranslation("trC2", -3.39, -152., -0.15);
  trC2->RegisterYourself();
  TGeoTranslation* trC3 = new TGeoTranslation("trC3", -3.39, +8.5, -0.15);
  trC3->RegisterYourself();
  TGeoTranslation* trC4 = new TGeoTranslation("trC4", -3.39, +151.8, -0.15);
  trC4->RegisterYourself();
  TGeoTranslation* trC5 = new TGeoTranslation("trC5", -3.39, 286.6, -0.15);
  trC5->RegisterYourself();

  TGeoCompositeShape* TOFrail =
    new TGeoCompositeShape("TOFrail", "(RaV1+RaO1:trRaO1)+C1:trC1+C2:trC2+C3:trC3+C4:trC4+C5:trC5");

  char nameTR[16];
  snprintf(nameTR, 16, "VolTOFrail");
  TGeoVolume* VolTOFrail = new TGeoVolume(nameTR, TOFrail, gGeoManager->GetMedium("FRAME_Aluminum"));
  VolTOFrail->SetName(nameTR);
  gGeoManager->GetVolume(nameTR)->SetVisibility(true);
  AliMatrix(idrotm[2102], 90.0, 180.0, 90.0, 270.0, 0.0, 180.0);

  for (i = 0; i < 18; i++) {
    char nameMo[16];
    snprintf(nameMo, 16, "BSEGMO%d", i);
    vmc->Gspos("VolTOFrail", 2 * i, nameMo, -66.27, 0., +56.33 + zsh, 0, "ONLY");
    vmc->Gspos("VolTOFrail", 2 * i + 1, nameMo, 66.27, 0., +56.33 + zsh, idrotm[2102], "ONLY");
  }

  //
  //    Geometry of Rails starts here
  //
  //
  //
  //    Rails for space-frame
  //
  float rbox[3];

  rbox[0] = 25.00;
  rbox[1] = 27.50;
  rbox[2] = 600.00;
  vmc->Gsvolu("BRS1", "BOX", kAir, rbox, 3);

  rbox[0] = 25.00;
  rbox[1] = 3.75;
  vmc->Gsvolu("BRS2", "BOX", kSteel, rbox, 3);

  rbox[0] = 3.00;
  rbox[1] = 20.00;
  vmc->Gsvolu("BRS3", "BOX", kSteel, rbox, 3);

  vmc->Gspos("BRS2", 1, "BRS1", 0., -27.5 + 3.75, 0., 0, "ONLY");
  vmc->Gspos("BRS2", 2, "BRS1", 0., 27.5 - 3.75, 0., 0, "ONLY");
  vmc->Gspos("BRS3", 1, "BRS1", 0., 0., 0., 0, "ONLY");
  vmc->Gspos("BRS1", 1, TOPNAME, -430. - 3., -190., 0., 0, "ONLY");
  vmc->Gspos("BRS1", 2, TOPNAME, 430. + 3., -190., 0., 0, "ONLY");

  rbox[0] = 3.0;
  rbox[1] = 145. / 4.;
  rbox[2] = 25.0;
  vmc->Gsvolu("BRS4", "BOX", kSteel, rbox, 3);

  vmc->Gspos("BRS4", 1, TOPNAME, 430. + 3., -190. + 55. / 2. + rbox[1], 224., 0, "ONLY");
  vmc->Gspos("BRS4", 2, TOPNAME, 430. + 3., -190. + 55. / 2. + rbox[1], -224., 0, "ONLY");

  //
  // The Backframe
  //
  // Inner radius
  float kBFMRin = 270.0;
  // Outer Radius
  float kBFMRou = 417.5;
  // Width
  float kBFMdz = 118.0;
  //
  //
  // Rings
  float kBFRdr = 7.5;
  float kBFRdz = 8.0;
  //
  //
  // Bars and Spokes
  //
  float kBFBd = 8.0;
  float kBFBdd = 0.6;

  // The Mother volume
  float tpar[3];
  tpar[0] = kBFMRin;
  tpar[1] = kBFMRou;
  tpar[2] = kBFMdz / 2.;
  vmc->Gsvolu("BFMO", "TUBE", kAir, tpar, 3);

  // CBL ////////////////////////////////////////////////////////
  //
  // TRD mother volume
  //

  ptrd1[0] = 47.4405 - 0.3;
  ptrd1[1] = 61.1765 - 0.3;
  ptrd1[2] = kBFMdz / 2.;
  ptrd1[3] = 38.95;
  vmc->Gsvolu("BFTRD", "TRD1", kAir, ptrd1, 4);
  gGeoManager->GetVolume("BFTRD")->SetVisibility(false);

  for (i = 0; i < 18; i++) {
    float phiBF = i * 20.0;
    dx = TMath::Sin(phiBF * kdeg2rad) * (342.0 - 12.62);
    dy = -TMath::Cos(phiBF * kdeg2rad) * (342.0 - 12.62);
    vmc->Gspos("BFTRD", i, "BFMO", dx, dy, 0.0, idrotm[2034 + i], "ONLY");
  }

  // CBL ////////////////////////////////////////////////////////

  // Rings
  //
  // Inner Ring
  tpar[0] = kBFMRin;
  tpar[1] = tpar[0] + kBFRdr;
  tpar[2] = kBFRdz / 2.;

  vmc->Gsvolu("BFIR", "TUBE", kSteel, tpar, 3);

  tpar[0] = tpar[0] + kBFBdd;
  tpar[1] = tpar[1] - kBFBdd;
  tpar[2] = (kBFRdz - 2. * kBFBdd) / 2.;

  vmc->Gsvolu("BFII", "TUBE", kAir, tpar, 3);
  vmc->Gspos("BFII", 1, "BFIR", 0., 0., 0., 0, "ONLY");

  //
  // Outer RING
  tpar[0] = kBFMRou - kBFRdr + 0.1;
  tpar[1] = kBFMRou;
  tpar[2] = kBFRdz / 2.;

  vmc->Gsvolu("BFOR", "TUBE", kSteel, tpar, 3);

  tpar[0] = tpar[0] + kBFBdd;
  tpar[1] = tpar[1] - kBFBdd;
  tpar[2] = (kBFRdz - 2. * kBFBdd) / 2.;

  vmc->Gsvolu("BFOO", "TUBE", kAir, tpar, 3);
  vmc->Gspos("BFOO", 1, "BFOR", 0., 0., 0., 0, "ONLY");

  dz = kBFMdz / 2. - kBFRdz / 2.;
  vmc->Gspos("BFIR", 1, "BFMO", 0., 0., dz, 0, "ONLY");
  vmc->Gspos("BFIR", 2, "BFMO", 0., 0., -dz, 0, "ONLY");
  vmc->Gspos("BFOR", 1, "BFMO", 0., 0., dz, 0, "ONLY");
  vmc->Gspos("BFOR", 2, "BFMO", 0., 0., -dz, 0, "ONLY");

  //
  // Longitudinal Bars
  //
  float bpar[3];

  bpar[0] = kBFBd / 2;
  bpar[1] = bpar[0];
  bpar[2] = kBFMdz / 2. - kBFBd;
  vmc->Gsvolu("BFLB", "BOX ", kSteel, bpar, 3);

  bpar[0] = bpar[0] - kBFBdd;
  bpar[1] = bpar[1] - kBFBdd;
  bpar[2] = bpar[2] - kBFBdd;
  vmc->Gsvolu("BFLL", "BOX ", kAir, bpar, 3);
  vmc->Gspos("BFLL", 1, "BFLB", 0., 0., 0., 0, "ONLY");

  for (i = 0; i < 18; i++) {
    float ro = kBFMRou - kBFBd / 2. - 0.02;
    float ri = kBFMRin + kBFBd / 2.;

    float phi0 = float(i) * 20.;

    float xb = ri * TMath::Cos(phi0 * kDegrad);
    float yb = ri * TMath::Sin(phi0 * kDegrad);
    AliMatrix(idrotm[2090 + i], 90.0, phi0, 90.0, phi0 + 270., 0., 0.);

    vmc->Gspos("BFLB", i + 1, "BFMO", xb, yb, 0., idrotm[2090 + i], "ONLY");

    xb = ro * TMath::Cos(phi0 * kDegrad);
    yb = ro * TMath::Sin(phi0 * kDegrad);

    vmc->Gspos("BFLB", i + 19, "BFMO", xb, yb, 0., idrotm[2090 + i], "ONLY");
  }

  //
  // Radial Bars
  //
  bpar[0] = (kBFMRou - kBFMRin - 2. * kBFRdr) / 2.;
  bpar[1] = kBFBd / 2;
  bpar[2] = bpar[1];
  //
  // Avoid overlap with circle
  float rr = kBFMRou - kBFRdr;
  float delta = rr - TMath::Sqrt(rr * rr - kBFBd * kBFBd / 4.) + 0.01;
  bpar[0] -= delta / 2.;

  vmc->Gsvolu("BFRB", "BOX ", kSteel, bpar, 3);

  bpar[0] = bpar[0] - kBFBdd;
  bpar[1] = bpar[1] - kBFBdd;
  bpar[2] = bpar[2] - kBFBdd;
  vmc->Gsvolu("BFRR", "BOX ", kAir, bpar, 3);
  vmc->Gspos("BFRR", 1, "BFRB", 0., 0., 0., 0, "ONLY");

  int iphi[10] = {0, 1, 3, 6, 8, 9, 10, 12, 15, 17};

  for (i = 0; i < 10; i++) {
    float rb = (kBFMRin + kBFMRou) / 2.;
    float phib = float(iphi[i]) * 20.;

    float xb = rb * TMath::Cos(phib * kDegrad);
    float yb = rb * TMath::Sin(phib * kDegrad);

    vmc->Gspos("BFRB", i + 1, "BFMO", xb, yb, dz, idrotm[2034 + iphi[i]], "ONLY");
    vmc->Gspos("BFRB", i + 11, "BFMO", xb, yb, -dz, idrotm[2034 + iphi[i]], "ONLY");
  }

  vmc->Gspos("BFMO", i + 19, TOPNAME, 0, 0, -376. - kBFMdz / 2. - 0.5, 0, "ONLY");

  //
  //
  //  The Baby Frame
  //
  //
  //
  // Inner radius
  float kBBMRin = 278.0;
  // Outer Radius
  float kBBMRou = 410.5;
  // Width
  float kBBMdz = 223.0;
  float kBBBdz = 6.0;
  float kBBBdd = 0.6;

  // The Mother volume

  ppgon[0] = 0.;
  ppgon[1] = 360.;
  ppgon[2] = 18.;

  ppgon[3] = 2.;
  ppgon[4] = -kBBMdz / 2.;
  ppgon[5] = kBBMRin;
  ppgon[6] = kBBMRou;

  ppgon[7] = -ppgon[4];
  ppgon[8] = ppgon[5];
  ppgon[9] = ppgon[6];

  vmc->Gsvolu("BBMO", "PGON", kAir, ppgon, 10);
  vmc->Gsdvn("BBCE", "BBMO", 18, 2);

  // CBL ////////////////////////////////////////////////////////
  //
  // TRD mother volume
  //

  AliMatrix(idrotm[2092], 90.0, 90.0, 0.0, 0.0, 90.0, 0.0);

  ptrd1[0] = 47.4405 - 2.5;
  ptrd1[1] = 61.1765 - 2.5;
  ptrd1[2] = kBBMdz / 2.;
  ptrd1[3] = 38.95;
  vmc->Gsvolu("BBTRD", "TRD1", kAir, ptrd1, 4);
  gGeoManager->GetVolume("BBTRD")->SetVisibility(false);
  vmc->Gspos("BBTRD", 1, "BBCE", 342.0 - 12.62, 0.0, 0.0, idrotm[2092], "ONLY");

  // CBL ////////////////////////////////////////////////////////

  // Longitudinal bars
  bpar[0] = kBBBdz / 2.;
  bpar[1] = bpar[0];
  bpar[2] = kBBMdz / 2. - kBBBdz;
  vmc->Gsvolu("BBLB", "BOX ", kSteel, bpar, 3);
  bpar[0] -= kBBBdd;
  bpar[1] -= kBBBdd;
  bpar[2] -= kBBBdd;
  vmc->Gsvolu("BBLL", "BOX ", kAir, bpar, 3);
  vmc->Gspos("BBLL", 1, "BBLB", 0., 0., 0., 0, "ONLY");

  dx = kBBMRin + kBBBdz / 2. + (bpar[1] + kBBBdd) * TMath::Sin(10. * kDegrad);
  dy = dx * TMath::Tan(10. * kDegrad) - kBBBdz / 2. / TMath::Cos(10. * kDegrad);
  vmc->Gspos("BBLB", 1, "BBCE", dx, dy, 0., idrotm[2052], "ONLY");

  dx = kBBMRou - kBBBdz / 2. - (bpar[1] + kBBBdd) * TMath::Sin(10. * kDegrad);
  dy = dx * TMath::Tan(10. * kDegrad) - kBBBdz / 2. / TMath::Cos(10. * kDegrad);

  vmc->Gspos("BBLB", 2, "BBCE", dx, dy, 0., idrotm[2052], "ONLY");

  //
  // Radial Bars
  //
  bpar[0] = (kBBMRou - kBBMRin) / 2. - kBBBdz;
  bpar[1] = kBBBdz / 2;
  bpar[2] = bpar[1];

  vmc->Gsvolu("BBRB", "BOX ", kSteel, bpar, 3);
  bpar[0] -= kBBBdd;
  bpar[1] -= kBBBdd;
  bpar[2] -= kBBBdd;
  vmc->Gsvolu("BBRR", "BOX ", kAir, bpar, 3);
  vmc->Gspos("BBRR", 1, "BBRB", 0., 0., 0., 0, "ONLY");

  dx = (kBBMRou + kBBMRin) / 2.;
  dy = ((kBBMRou + kBBMRin) / 2) * TMath::Tan(10 * kDegrad) - kBBBdz / 2. / TMath::Cos(10 * kDegrad);
  dz = kBBMdz / 2. - kBBBdz / 2.;

  vmc->Gspos("BBRB", 1, "BBCE", dx, dy, dz, idrotm[2052], "ONLY");
  vmc->Gspos("BBRB", 2, "BBCE", dx, dy, -dz, idrotm[2052], "ONLY");
  vmc->Gspos("BBRB", 3, "BBCE", dx, dy, 0., idrotm[2052], "ONLY");

  //
  // Circular bars
  //
  //  Inner

  bpar[1] = kBBMRin * TMath::Sin(10. * kDegrad);
  bpar[0] = kBBBdz / 2;
  bpar[2] = bpar[0];
  vmc->Gsvolu("BBC1", "BOX ", kSteel, bpar, 3);
  bpar[0] -= kBBBdd;
  bpar[1] -= kBBBdd;
  bpar[2] -= kBBBdd;
  vmc->Gsvolu("BBC2", "BOX ", kAir, bpar, 3);
  vmc->Gspos("BBC2", 1, "BBC1", 0., 0., 0., 0, "ONLY");
  dx = kBBMRin + kBBBdz / 2;
  dy = 0.;
  vmc->Gspos("BBC1", 1, "BBCE", dx, dy, dz, 0, "ONLY");
  vmc->Gspos("BBC1", 2, "BBCE", dx, dy, -dz, 0, "ONLY");
  //
  // Outer
  bpar[1] = (kBBMRou - kBBBdz) * TMath::Sin(10. * kDegrad);
  bpar[0] = kBBBdz / 2;
  bpar[2] = bpar[0];
  vmc->Gsvolu("BBC3", "BOX ", kSteel, bpar, 3);
  bpar[0] -= kBBBdd;
  bpar[1] -= kBBBdd;
  bpar[2] -= kBBBdd;
  vmc->Gsvolu("BBC4", "BOX ", kAir, bpar, 3);
  vmc->Gspos("BBC4", 1, "BBC3", 0., 0., 0., 0, "ONLY");
  dx = kBBMRou - kBBBdz / 2;
  dy = 0.;
  vmc->Gspos("BBC3", 1, "BBCE", dx, dy, dz, 0, "ONLY");
  vmc->Gspos("BBC3", 2, "BBCE", dx, dy, -dz, 0, "ONLY");
  //
  // Diagonal Bars
  //
  h = (kBBMRou - kBBMRin - 2. * kBBBdz);
  ;
  d = kBBBdz;
  dz = kBBMdz / 2. - 1.6 * kBBBdz;
  dq = h * h + dz * dz;

  x = TMath::Sqrt((dz * dz - d * d) / dq + d * d * h * h / dq / dq) + d * h / dq;

  theta = kRaddeg * TMath::ACos(x);

  ptrap[0] = dz / 2.;
  ptrap[1] = theta;
  ptrap[2] = 0.;
  ptrap[3] = d / 2;
  ptrap[4] = d / x / 2;
  ptrap[5] = ptrap[4];
  ptrap[6] = 0;
  ptrap[7] = ptrap[3];
  ptrap[8] = ptrap[4];
  ptrap[9] = ptrap[4];
  ptrap[10] = 0;
  vmc->Gsvolu("BBD1", "TRAP", kSteel, ptrap, 11);
  ptrap[3] = d / 2 - kBBBdd;
  ptrap[4] = (d / 2 - kBBBdd) / x;
  ptrap[5] = ptrap[4];
  ptrap[7] = ptrap[3];
  ptrap[8] = ptrap[4];
  ptrap[9] = ptrap[4];
  vmc->Gsvolu("BBD3", "TRAP", kAir, ptrap, 11);
  vmc->Gspos("BBD3", 1, "BBD1", 0.0, 0.0, 0., 0, "ONLY");
  dx = (kBBMRou + kBBMRin) / 2.;
  dy = ((kBBMRou + kBBMRin) / 2) * TMath::Tan(10 * kDegrad) - kBBBdz / 2. / TMath::Cos(10 * kDegrad);
  vmc->Gspos("BBD1", 1, "BBCE", dx, dy, dz / 2. + kBBBdz / 2., idrotm[2052], "ONLY");

  ptrap[0] = dz / 2.;
  ptrap[1] = -theta;
  ptrap[2] = 0.;
  ptrap[3] = d / 2;
  ptrap[4] = d / 2 / x;
  ptrap[5] = ptrap[4];
  ptrap[6] = 0;
  ptrap[7] = ptrap[3];
  ptrap[8] = ptrap[4];
  ptrap[9] = ptrap[4];
  ptrap[10] = 0;
  vmc->Gsvolu("BBD2", "TRAP", kSteel, ptrap, 11);
  ptrap[3] = d / 2 - kBBBdd;
  ptrap[4] = (d / 2 - kBBBdd) / x;
  ptrap[5] = ptrap[4];
  ptrap[7] = ptrap[3];
  ptrap[8] = ptrap[4];
  ptrap[9] = ptrap[4];
  vmc->Gsvolu("BBD4", "TRAP", kAir, ptrap, 11);
  vmc->Gspos("BBD4", 1, "BBD2", 0.0, 0.0, 0., 0, "ONLY");
  dx = (kBBMRou + kBBMRin) / 2.;
  dy = ((kBBMRou + kBBMRin) / 2) * TMath::Tan(10 * kDegrad) - kBBBdz / 2. / TMath::Cos(10 * kDegrad);
  vmc->Gspos("BBD2", 1, "BBCE", dx, dy, -dz / 2. - kBBBdz / 2., idrotm[2052], "ONLY");

  vmc->Gspos("BBMO", 1, TOPNAME, 0., 0., +376. + kBBMdz / 2. + 0.5, 0, "ONLY");
}
} // namespace passive
} // namespace o2
